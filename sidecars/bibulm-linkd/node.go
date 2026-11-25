package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"tailscale.com/client/local"
	"tailscale.com/ipn"
	"tailscale.com/tailcfg"
	"tailscale.com/tsnet"
)

type NodeStatus struct {
	State             string `json:"state"` // starting | needs_login | connecting | connected | error
	AuthURL           string `json:"auth_url,omitempty"`
	Hostname          string `json:"hostname"`
	NodeID            string `json:"node_id,omitempty"`
	TailscaleIP       string `json:"tailscale_ip,omitempty"`
	MagicDNS          string `json:"magic_dns,omitempty"`
	TailnetName       string `json:"tailnet_name,omitempty"`
	AccountLogin      string `json:"account_login,omitempty"`
	AccountName       string `json:"account_name,omitempty"`
	AccessPolicy      string `json:"access_policy"`
	ExposeLocalModels bool   `json:"expose_local_models"`
	Error             string `json:"error,omitempty"`
}

type LinkNode struct {
	cfg    Config
	cancel context.CancelFunc
	srv    *tsnet.Server
	lc     *local.Client

	mu                sync.RWMutex
	status            NodeStatus
	selfUserID        tailcfg.UserID
	gatewayPort       int
	accessPolicy      string
	exposeLocalModels bool
	meshServer        *http.Server
	meshStarted       bool
	peers             []DiscoveredPeer
	peersMu           sync.RWMutex

	dialClient *http.Client
}

func NewLinkNode(ctx context.Context, cfg Config, cancel context.CancelFunc) (*LinkNode, error) {
	if err := os.MkdirAll(cfg.StateDir, 0o700); err != nil {
		return nil, fmt.Errorf("create state dir: %w", err)
	}

	srv := &tsnet.Server{
		Hostname:   cfg.Hostname,
		Dir:        cfg.StateDir,
		ControlURL: cfg.ControlURL,
		AuthKey:    cfg.AuthKey,
		Logf: func(format string, args ...any) {
			// Suppress noisy tsnet debug output unless explicitly enabled.
			if os.Getenv("BIBULM_LINK_DEBUG") == "1" {
				log.Printf("[tsnet] "+format, args...)
			}
		},
	}

	n := &LinkNode{
		cfg:               cfg,
		cancel:            cancel,
		srv:               srv,
		gatewayPort:       cfg.GatewayPort,
		accessPolicy:      normalizePolicy(cfg.AccessPolicy),
		exposeLocalModels: cfg.ExposeLocalModels,
		status: NodeStatus{
			State:             "starting",
			Hostname:          cfg.Hostname,
			AccessPolicy:      normalizePolicy(cfg.AccessPolicy),
			ExposeLocalModels: cfg.ExposeLocalModels,
		},
	}

	n.dialClient = &http.Client{
		Transport: &http.Transport{
			DialContext:           srv.Dial,
			MaxIdleConnsPerHost:   16,
			IdleConnTimeout:       60 * time.Second,
			ResponseHeaderTimeout: 0, // Allow long LLM prefill / JIT model load
		},
		Timeout: 0, // Streaming SSE responses must not time out
	}

	return n, nil
}

func normalizePolicy(p string) string {
	if p == "same_tailnet" {
		return "same_tailnet"
	}
	return "self_only"
}

func (n *LinkNode) Close() {
	n.mu.Lock()
	ms := n.meshServer
	n.meshServer = nil
	n.mu.Unlock()
	if ms != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_ = ms.Shutdown(ctx)
		cancel()
	}
	if n.srv != nil {
		_ = n.srv.Close()
	}
}

func (n *LinkNode) Run(ctx context.Context) {
	if err := n.srv.Start(); err != nil {
		n.setError(fmt.Sprintf("failed to start tsnet: %v", err))
		return
	}

	lc, err := n.srv.LocalClient()
	if err != nil {
		n.setError(fmt.Sprintf("failed to get tsnet local client: %v", err))
		return
	}
	n.mu.Lock()
	n.lc = lc
	n.mu.Unlock()

	go n.watchBus(ctx, lc)

	// Periodic status + peer discovery loop
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	// Initial poll
	n.syncStatusFromClient(ctx, lc)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			n.syncStatusFromClient(ctx, lc)
			if n.isConnected() {
				n.RefreshPeers(ctx)
			}
		}
	}
}

func (n *LinkNode) watchBus(ctx context.Context, lc *local.Client) {
	watcher, err := lc.WatchIPNBus(ctx, ipn.NotifyInitialState)
	if err != nil {
		log.Printf("WatchIPNBus failed: %v", err)
		return
	}
	defer watcher.Close()

	loginRequested := false
	for {
		notify, err := watcher.Next()
		if err != nil {
			if ctx.Err() == nil {
				log.Printf("WatchIPNBus closed: %v", err)
			}
			return
		}

		if notify.BrowseToURL != nil && *notify.BrowseToURL != "" {
			n.mu.Lock()
			n.status.State = "needs_login"
			n.status.AuthURL = *notify.BrowseToURL
			n.status.Error = ""
			n.mu.Unlock()
		}

		if notify.State != nil {
			switch *notify.State {
			case ipn.NeedsLogin:
				n.mu.Lock()
				n.status.State = "needs_login"
				n.mu.Unlock()
				if !loginRequested {
					loginRequested = true
					go func() {
						if err := lc.StartLoginInteractive(ctx); err != nil && ctx.Err() == nil {
							log.Printf("StartLoginInteractive error: %v", err)
						}
					}()
				}
			case ipn.Starting:
				n.mu.Lock()
				if n.status.State != "needs_login" {
					n.status.State = "connecting"
				}
				n.mu.Unlock()
			case ipn.Running:
				loginRequested = false
				n.syncStatusFromClient(ctx, lc)
				n.ensureMeshListener(ctx)
				go n.RefreshPeers(ctx)
			}
		}
	}
}

func (n *LinkNode) syncStatusFromClient(ctx context.Context, lc *local.Client) {
	st, err := lc.Status(ctx)
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("lc.Status error: %v", err)
		}
		return
	}

	n.mu.Lock()
	defer n.mu.Unlock()

	switch st.BackendState {
	case "Running":
		n.status.State = "connected"
		n.status.AuthURL = ""
		n.status.Error = ""
	case "NeedsLogin":
		n.status.State = "needs_login"
		if st.AuthURL != "" {
			n.status.AuthURL = st.AuthURL
		}
	case "Starting":
		if n.status.State != "needs_login" {
			n.status.State = "connecting"
		}
	}

	if st.CurrentTailnet != nil {
		n.status.TailnetName = st.CurrentTailnet.Name
	}

	if st.Self != nil {
		n.status.NodeID = string(st.Self.ID)
		if len(st.Self.TailscaleIPs) > 0 {
			n.status.TailscaleIP = st.Self.TailscaleIPs[0].String()
		}
		n.status.MagicDNS = st.Self.DNSName
		n.selfUserID = st.Self.UserID
		if profile, ok := st.User[st.Self.UserID]; ok {
			n.status.AccountLogin = profile.LoginName
			n.status.AccountName = profile.DisplayName
		}
	}

	n.status.AccessPolicy = n.accessPolicy
	n.status.ExposeLocalModels = n.exposeLocalModels
}

func (n *LinkNode) ensureMeshListener(ctx context.Context) {
	n.mu.Lock()
	if n.meshStarted {
		n.mu.Unlock()
		return
	}
	n.meshStarted = true
	n.mu.Unlock()

	addr := fmt.Sprintf(":%d", n.cfg.MeshPort)
	ln, err := n.srv.Listen("tcp", addr)
	if err != nil {
		log.Printf("tsnet Listen(%s) failed: %v", addr, err)
		n.mu.Lock()
		n.meshStarted = false
		n.mu.Unlock()
		return
	}

	ms := &http.Server{
		Handler:           n.meshHandler(),
		ReadHeaderTimeout: 15 * time.Second,
	}
	n.mu.Lock()
	n.meshServer = ms
	n.mu.Unlock()

	go func() {
		log.Printf("tailnet mesh listener active on %s", addr)
		if err := ms.Serve(ln); err != nil && err != http.ErrServerClosed {
			log.Printf("mesh server error: %v", err)
		}
	}()

	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()
}

func (n *LinkNode) isConnected() bool {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.status.State == "connected"
}

func (n *LinkNode) Status() NodeStatus {
	n.mu.RLock()
	defer n.mu.RUnlock()
	return n.status
}

func (n *LinkNode) setError(msg string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.status.State = "error"
	n.status.Error = msg
}

func checkAccessPolicy(policy string, selfUserID, callerUserID tailcfg.UserID) bool {
	if policy == "same_tailnet" {
		return true
	}
	// Default strict mode: only devices belonging to the exact same Tailscale user account.
	return selfUserID != 0 && callerUserID == selfUserID
}

func (n *LinkNode) BridgeHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /bridge/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, n.Status())
	})
	mux.HandleFunc("GET /bridge/peers", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, n.Peers())
	})
	mux.HandleFunc("POST /bridge/peers/refresh", func(w http.ResponseWriter, r *http.Request) {
		n.RefreshPeers(r.Context())
		writeJSON(w, http.StatusOK, n.Peers())
	})
	mux.HandleFunc("POST /bridge/config", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			GatewayPort       *int    `json:"gateway_port"`
			AccessPolicy      *string `json:"access_policy"`
			ExposeLocalModels *bool   `json:"expose_local_models"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		n.mu.Lock()
		if req.GatewayPort != nil && *req.GatewayPort > 0 {
			n.gatewayPort = *req.GatewayPort
		}
		if req.AccessPolicy != nil {
			n.accessPolicy = normalizePolicy(*req.AccessPolicy)
			n.status.AccessPolicy = n.accessPolicy
		}
		if req.ExposeLocalModels != nil {
			n.exposeLocalModels = *req.ExposeLocalModels
			n.status.ExposeLocalModels = *req.ExposeLocalModels
		}
		st := n.status
		n.mu.Unlock()
		writeJSON(w, http.StatusOK, st)
	})
	mux.HandleFunc("POST /bridge/logout", func(w http.ResponseWriter, r *http.Request) {
		n.mu.RLock()
		lc := n.lc
		n.mu.RUnlock()
		if lc != nil {
			logoutCtx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
			defer cancel()
			if err := lc.Logout(logoutCtx); err != nil {
				http.Error(w, fmt.Sprintf("logout failed: %v", err), http.StatusInternalServerError)
				return
			}
		}
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
		go func() {
			time.Sleep(150 * time.Millisecond)
			n.cancel()
		}()
	})
	mux.HandleFunc("POST /bridge/shutdown", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
		go func() {
			time.Sleep(100 * time.Millisecond)
			n.cancel()
		}()
	})
	mux.HandleFunc("/bridge/proxy/", n.handleBridgeProxy)
	return mux
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// Helper to verify IP strings cleanly when routing proxy calls.
func isValidTailscaleIP(ip string) bool {
	parsed := net.ParseIP(ip)
	return parsed != nil
}
