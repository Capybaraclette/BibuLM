package main

import (
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

// meshHandler handles incoming requests from other peers on the Tailnet (:11435),
// authenticates them via Tailscale WhoIs, enforces local access policy, and proxies
// them to the local BibuLM ApiGateway with unbuffered SSE streaming.
func (n *LinkNode) meshHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.mu.RLock()
		lc := n.lc
		selfUID := n.selfUserID
		policy := n.accessPolicy
		expose := n.exposeLocalModels
		gwPort := n.gatewayPort
		n.mu.RUnlock()

		if lc == nil {
			http.Error(w, "tailnet node not ready", http.StatusServiceUnavailable)
			return
		}

		who, err := lc.WhoIs(r.Context(), r.RemoteAddr)
		if err != nil || who == nil || who.UserProfile == nil {
			http.Error(w, "tailscale identity verification failed", http.StatusForbidden)
			return
		}

		if !checkAccessPolicy(policy, selfUID, who.UserProfile.ID) {
			http.Error(w, "access denied by BibuLM Link policy", http.StatusForbidden)
			return
		}

		// Allow /link/v1/info and /link/v1/notify even when expose_local_models is false
		// so peers can see device status and receive state-change notifications.
		if !expose && r.URL.Path != "/link/v1/info" && r.URL.Path != "/link/v1/notify" {
			http.Error(w, "this device is in client-only mode and does not share local models", http.StatusForbidden)
			return
		}

		target, _ := url.Parse(fmt.Sprintf("http://127.0.0.1:%d", gwPort))
		rp := newStreamingReverseProxy(target, nil)
		origDirector := rp.Director
		rp.Director = func(req *http.Request) {
			origDirector(req)
			req.Header.Set("X-Link-Secret", n.cfg.Secret)
			req.Header.Set("X-Link-Peer-User", who.UserProfile.LoginName)
			if who.Node != nil {
				req.Header.Set("X-Link-Peer-Node", who.Node.ComputedName)
			}
		}
		rp.ServeHTTP(w, r)
	})
}

// handleBridgeProxy proxies outgoing requests from BibuLM (Rust) to a remote Tailnet peer
// over tsnet.Server.Dial: /bridge/proxy/{peer_ip}/{remote_path...}
func (n *LinkNode) handleBridgeProxy(w http.ResponseWriter, r *http.Request) {
	trimmed := strings.TrimPrefix(r.URL.Path, "/bridge/proxy/")
	parts := strings.SplitN(trimmed, "/", 2)
	if len(parts) < 1 || !isValidTailscaleIP(parts[0]) {
		http.Error(w, "invalid peer IP in proxy path", http.StatusBadRequest)
		return
	}
	peerIP := parts[0]
	remotePath := "/"
	if len(parts) == 2 {
		remotePath = "/" + parts[1]
	}

	target, _ := url.Parse(fmt.Sprintf("http://%s:%d", peerIP, n.cfg.MeshPort))
	rp := newStreamingReverseProxy(target, n.dialClient.Transport)
	origDirector := rp.Director
	rp.Director = func(req *http.Request) {
		origDirector(req)
		req.URL.Path = remotePath
		req.URL.RawPath = ""
		req.Header.Del("X-Link-Secret")
	}
	rp.ServeHTTP(w, r)
}

func newStreamingReverseProxy(target *url.URL, transport http.RoundTripper) *httputil.ReverseProxy {
	rp := httputil.NewSingleHostReverseProxy(target)
	// FlushInterval -1 guarantees immediate flushing of SSE token streams.
	rp.FlushInterval = -1
	if transport != nil {
		rp.Transport = transport
	}
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		http.Error(w, fmt.Sprintf("BibuLM Link upstream error: %v", err), http.StatusBadGateway)
	}
	return rp
}
