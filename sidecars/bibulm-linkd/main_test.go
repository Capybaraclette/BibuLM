package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"tailscale.com/tailcfg"
)

func TestSanitizeHostname(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"PC-Salon", "bibulm-pc-salon"},
		{"bibulm-My_Rig 4090!", "bibulm-my-rig-4090"},
		{"   ", "bibulm-node"},
		{"MacBook Pro (M4)", "bibulm-macbook-pro-m4"},
	}
	for _, tc := range cases {
		if got := sanitizeHostname(tc.in); got != tc.want {
			t.Errorf("sanitizeHostname(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestBridgeAuthMiddleware(t *testing.T) {
	okHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := bridgeAuthMiddleware("super-secret", okHandler)

	req := httptest.NewRequest(http.MethodGet, "/bridge/status", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without secret, got %d", rec.Code)
	}

	req2 := httptest.NewRequest(http.MethodGet, "/bridge/status", nil)
	req2.Header.Set("X-Link-Secret", "wrong")
	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 with wrong secret, got %d", rec2.Code)
	}

	req3 := httptest.NewRequest(http.MethodGet, "/bridge/status", nil)
	req3.Header.Set("X-Link-Secret", "super-secret")
	rec3 := httptest.NewRecorder()
	h.ServeHTTP(rec3, req3)
	if rec3.Code != http.StatusOK {
		t.Fatalf("expected 200 with valid secret, got %d", rec3.Code)
	}
}

func TestCheckAccessPolicy(t *testing.T) {
	selfID := tailcfg.UserID(42)
	otherID := tailcfg.UserID(99)

	if !checkAccessPolicy("self_only", selfID, selfID) {
		t.Fatal("self_only must allow same user")
	}
	if checkAccessPolicy("self_only", selfID, otherID) {
		t.Fatal("self_only must deny other user")
	}
	if !checkAccessPolicy("same_tailnet", selfID, otherID) {
		t.Fatal("same_tailnet must allow other tailnet user")
	}
}

func TestStreamingReverseProxySSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for i := 0; i < 3; i++ {
			fmt.Fprintf(w, "data: chunk-%d\n\n", i)
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	u, _ := url.Parse(upstream.URL)
	proxy := httptest.NewServer(newStreamingReverseProxy(u, nil))
	defer proxy.Close()

	resp, err := http.Get(proxy.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("proxy GET failed: %v", err)
	}
	defer resp.Body.Close()

	buf := make([]byte, 256)
	n, _ := resp.Body.Read(buf)
	body := string(buf[:n])
	if !strings.Contains(body, "data: chunk-0") {
		t.Fatalf("expected SSE chunk, got %q", body)
	}
}

func TestPeerInfoPayloadJSONCompatibility(t *testing.T) {
	legacy := `{"device_name":"PC-Salon","app_version":"0.4.0","expose_local_models":true,"allow_remote_load":true,"gpu_summary":"RTX 4090","vram_used_mib":1000,"vram_total_mib":24576,"ram_used_mib":8000,"ram_total_mib":65536,"loaded_models":[],"available_models":[]}`
	var p1 PeerInfoPayload
	if err := json.Unmarshal([]byte(legacy), &p1); err != nil {
		t.Fatalf("legacy PeerInfoPayload unmarshal failed: %v", err)
	}
	if p1.AllowRemoteManage || p1.IsHeadless || p1.RuntimesCount != 0 {
		t.Fatalf("expected zero defaults for omitted fields, got %+v", p1)
	}

	full := `{"device_name":"GPU-Server","app_version":"0.4.3","expose_local_models":true,"allow_remote_load":true,"allow_remote_manage":true,"is_headless":true,"platform":"linux-x86_64","runtimes_count":2,"active_runtime_tag":"b5200","active_downloads_count":1,"gpu_summary":"2x RTX 4090","vram_used_mib":2000,"vram_total_mib":49152,"ram_used_mib":16000,"ram_total_mib":131072,"loaded_models":[],"available_models":[]}`
	var p2 PeerInfoPayload
	if err := json.Unmarshal([]byte(full), &p2); err != nil {
		t.Fatalf("full PeerInfoPayload unmarshal failed: %v", err)
	}
	if !p2.AllowRemoteManage || !p2.IsHeadless || p2.Platform != "linux-x86_64" || p2.RuntimesCount != 2 || p2.ActiveRuntimeTag != "b5200" || p2.ActiveDownloadsCount != 1 {
		t.Fatalf("unexpected parsed fields: %+v", p2)
	}
	if _, err := json.Marshal(p2); err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
}
