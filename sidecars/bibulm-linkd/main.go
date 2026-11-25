package main

import (
	"bufio"
	"context"
	"crypto/subtle"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

type Config struct {
	StateDir          string
	Hostname          string
	ControlURL        string
	GatewayPort       int
	MeshPort          int
	AccessPolicy      string
	ExposeLocalModels bool
	Secret            string
	AuthKey           string
}

func main() {
	// Keep all diagnostic logs on stderr so stdout remains clean for the JSON handshake.
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	cfg := Config{}
	flag.StringVar(&cfg.StateDir, "state-dir", "", "Directory for persistent tsnet state")
	flag.StringVar(&cfg.Hostname, "hostname", "", "Tailnet hostname (will be prefixed with bibulm-)")
	flag.StringVar(&cfg.ControlURL, "control-url", "", "Optional custom Tailscale/Headscale control URL")
	flag.IntVar(&cfg.GatewayPort, "gateway-port", 1234, "Local Bibulm OpenAI gateway port")
	flag.IntVar(&cfg.MeshPort, "mesh-port", 11435, "TCP port to listen on inside the tailnet")
	flag.StringVar(&cfg.AccessPolicy, "access-policy", "self_only", "Access policy: self_only or same_tailnet")
	flag.BoolVar(&cfg.ExposeLocalModels, "expose-models", true, "Expose local models over the tailnet")
	flag.Parse()

	cfg.Secret = strings.TrimSpace(os.Getenv("BIBULM_LINK_SECRET"))
	if cfg.Secret == "" {
		log.Fatal("BIBULM_LINK_SECRET environment variable is required")
	}
	cfg.AuthKey = strings.TrimSpace(os.Getenv("TS_AUTHKEY"))
	if cfg.AuthKey == "" {
		cfg.AuthKey = strings.TrimSpace(os.Getenv("BIBULM_LINK_AUTHKEY"))
	}
	if cfg.StateDir == "" {
		log.Fatal("--state-dir is required")
	}
	if cfg.Hostname == "" {
		if h, err := os.Hostname(); err == nil && h != "" {
			cfg.Hostname = h
		} else {
			cfg.Hostname = "node"
		}
	}
	cfg.Hostname = sanitizeHostname(cfg.Hostname)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Exit cleanly if parent closes stdin (guarantees zero orphan sidecar processes).
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if strings.TrimSpace(scanner.Text()) == "shutdown" {
				cancel()
				return
			}
		}
		cancel()
	}()

	node, err := NewLinkNode(ctx, cfg, cancel)
	if err != nil {
		log.Fatalf("failed to initialize LinkNode: %v", err)
	}
	defer node.Close()

	bridgeLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("failed to listen on local bridge: %v", err)
	}
	bridgePort := bridgeLn.Addr().(*net.TCPAddr).Port

	bridgeServer := &http.Server{
		Handler:           bridgeAuthMiddleware(cfg.Secret, node.BridgeHandler()),
		ReadHeaderTimeout: 15 * time.Second,
	}

	go func() {
		if err := bridgeServer.Serve(bridgeLn); err != nil && err != http.ErrServerClosed {
			log.Printf("bridge server error: %v", err)
			cancel()
		}
	}()

	// Emit single-line JSON readiness handshake on stdout for Bibulm.
	handshake := map[string]any{
		"event":       "ready",
		"bridge_port": bridgePort,
		"hostname":    cfg.Hostname,
		"pid":         os.Getpid(),
	}
	line, _ := json.Marshal(handshake)
	fmt.Println(string(line))

	// Start tsnet node asynchronously so the local bridge is immediately queryable.
	go node.Run(ctx)

	<-ctx.Done()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	_ = bridgeServer.Shutdown(shutdownCtx)
}

func sanitizeHostname(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.TrimPrefix(s, "bibulm-")
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		isAlphaNum := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if isAlphaNum {
			b.WriteRune(r)
			lastDash = false
		} else if !lastDash && b.Len() > 0 {
			b.WriteByte('-')
			lastDash = true
		}
	}
	slug := strings.Trim(b.String(), "-")
	if slug == "" {
		slug = "node"
	}
	const prefix = "bibulm-"
	maxSlug := 63 - len(prefix)
	if len(slug) > maxSlug {
		slug = strings.TrimRight(slug[:maxSlug], "-")
	}
	return prefix + slug
}

func bridgeAuthMiddleware(secret string, next http.Handler) http.Handler {
	expected := []byte(secret)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := []byte(r.Header.Get("X-Link-Secret"))
		if len(got) != len(expected) || subtle.ConstantTimeCompare(got, expected) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}
