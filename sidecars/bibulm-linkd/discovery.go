package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

type RemoteModelSummary struct {
	ID                 string  `json:"id"`
	DisplayName        string  `json:"display_name"`
	Loaded             bool    `json:"loaded"`
	Quantization       *string `json:"quantization,omitempty"`
	Architecture       *string `json:"architecture,omitempty"`
	SizeBytes          uint64  `json:"size_bytes,omitempty"`
	MaxContextLength   *uint32 `json:"max_context_length,omitempty"`
	SupportsVision     bool    `json:"supports_vision"`
	SupportsReasoning  bool    `json:"supports_reasoning"`
	PlacementStatus    *string `json:"placement_status,omitempty"`
}

type PeerInfoPayload struct {
	DeviceName           string               `json:"device_name"`
	AppVersion           string               `json:"app_version"`
	ExposeLocalModels    bool                 `json:"expose_local_models"`
	AllowRemoteLoad      bool                 `json:"allow_remote_load"`
	AllowRemoteManage    bool                 `json:"allow_remote_manage"`
	IsHeadless           bool                 `json:"is_headless"`
	Platform             string               `json:"platform,omitempty"`
	RuntimesCount        int                  `json:"runtimes_count"`
	ActiveRuntimeTag     string               `json:"active_runtime_tag,omitempty"`
	ActiveDownloadsCount int                  `json:"active_downloads_count"`
	GpuSummary           string               `json:"gpu_summary"`
	VramUsedMiB          float64              `json:"vram_used_mib"`
	VramTotalMiB         float64              `json:"vram_total_mib"`
	RamUsedMiB           float64              `json:"ram_used_mib"`
	RamTotalMiB          float64              `json:"ram_total_mib"`
	LoadedModels         []RemoteModelSummary `json:"loaded_models"`
	AvailableModels      []RemoteModelSummary `json:"available_models"`
}

type DiscoveredPeer struct {
	NodeID               string               `json:"node_id"`
	Slug                 string               `json:"slug"`
	Hostname             string               `json:"hostname"`
	DeviceName           string               `json:"device_name"`
	TailscaleIP          string               `json:"tailscale_ip"`
	MagicDNS             string               `json:"magic_dns"`
	OS                   string               `json:"os"`
	Online               bool                 `json:"online"`
	ConnectionType       string               `json:"connection_type"` // direct | derp | unknown
	Relay                string               `json:"relay,omitempty"`
	LatencyMs            *uint64              `json:"latency_ms,omitempty"`
	AppVersion           string               `json:"app_version"`
	ExposeLocalModels    bool                 `json:"expose_local_models"`
	AllowRemoteLoad      bool                 `json:"allow_remote_load"`
	AllowRemoteManage    bool                 `json:"allow_remote_manage"`
	IsHeadless           bool                 `json:"is_headless"`
	Platform             string               `json:"platform,omitempty"`
	RuntimesCount        int                  `json:"runtimes_count"`
	ActiveRuntimeTag     string               `json:"active_runtime_tag,omitempty"`
	ActiveDownloadsCount int                  `json:"active_downloads_count"`
	GpuSummary           string               `json:"gpu_summary"`
	VramUsedMiB          float64              `json:"vram_used_mib"`
	VramTotalMiB         float64              `json:"vram_total_mib"`
	RamUsedMiB           float64              `json:"ram_used_mib"`
	RamTotalMiB          float64              `json:"ram_total_mib"`
	LoadedModels         []RemoteModelSummary `json:"loaded_models"`
	AvailableModels      []RemoteModelSummary `json:"available_models"`
}

func (n *LinkNode) Peers() []DiscoveredPeer {
	n.peersMu.RLock()
	defer n.peersMu.RUnlock()
	out := make([]DiscoveredPeer, len(n.peers))
	copy(out, n.peers)
	return out
}

func (n *LinkNode) RefreshPeers(ctx context.Context) {
	n.mu.RLock()
	lc := n.lc
	selfID := ""
	if n.status.NodeID != "" {
		selfID = n.status.NodeID
	}
	n.mu.RUnlock()

	if lc == nil {
		return
	}

	st, err := lc.Status(ctx)
	if err != nil {
		return
	}

	type candidate struct {
		nodeID   string
		hostname string
		slug     string
		ip       string
		dns      string
		osName   string
		connType string
		relay    string
	}

	var candidates []candidate
	for _, p := range st.Peer {
		if !p.Online || len(p.TailscaleIPs) == 0 {
			continue
		}
		if string(p.ID) == selfID {
			continue
		}
		hostLow := strings.ToLower(p.HostName)
		dnsLow := strings.ToLower(p.DNSName)
		if !strings.HasPrefix(hostLow, "bibulm-") && !strings.HasPrefix(dnsLow, "bibulm-") {
			continue
		}
		slug := strings.TrimPrefix(hostLow, "bibulm-")
		if slug == "" {
			slug = hostLow
		}
		connType := "unknown"
		if p.CurAddr != "" {
			connType = "direct"
		} else if p.Relay != "" {
			connType = "derp"
		}
		candidates = append(candidates, candidate{
			nodeID:   string(p.ID),
			hostname: p.HostName,
			slug:     slug,
			ip:       p.TailscaleIPs[0].String(),
			dns:      strings.TrimSuffix(p.DNSName, "."),
			osName:   p.OS,
			connType: connType,
			relay:    p.Relay,
		})
	}

	var (
		wg      sync.WaitGroup
		resMu   sync.Mutex
		results []DiscoveredPeer
	)

	sem := make(chan struct{}, 8)
	for _, c := range candidates {
		wg.Add(1)
		go func(c candidate) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()

			started := time.Now()
			info, err := n.probePeerInfo(probeCtx, c.ip)
			if err != nil {
				return
			}
			rtt := uint64(time.Since(started).Milliseconds())

			devName := strings.TrimSpace(info.DeviceName)
			if devName == "" {
				devName = c.slug
			}

			loaded := info.LoadedModels
			if loaded == nil {
				loaded = []RemoteModelSummary{}
			}
			avail := info.AvailableModels
			if avail == nil {
				avail = []RemoteModelSummary{}
			}

			resMu.Lock()
			results = append(results, DiscoveredPeer{
				NodeID:               c.nodeID,
				Slug:                 c.slug,
				Hostname:             c.hostname,
				DeviceName:           devName,
				TailscaleIP:          c.ip,
				MagicDNS:             c.dns,
				OS:                   c.osName,
				Online:               true,
				ConnectionType:       c.connType,
				Relay:                c.relay,
				LatencyMs:            &rtt,
				AppVersion:           info.AppVersion,
				ExposeLocalModels:    info.ExposeLocalModels,
				AllowRemoteLoad:      info.AllowRemoteLoad,
				AllowRemoteManage:    info.AllowRemoteManage,
				IsHeadless:           info.IsHeadless,
				Platform:             info.Platform,
				RuntimesCount:        info.RuntimesCount,
				ActiveRuntimeTag:     info.ActiveRuntimeTag,
				ActiveDownloadsCount: info.ActiveDownloadsCount,
				GpuSummary:           info.GpuSummary,
				VramUsedMiB:          info.VramUsedMiB,
				VramTotalMiB:         info.VramTotalMiB,
				RamUsedMiB:           info.RamUsedMiB,
				RamTotalMiB:          info.RamTotalMiB,
				LoadedModels:         loaded,
				AvailableModels:      avail,
			})
			resMu.Unlock()
		}(c)
	}
	wg.Wait()

	sort.Slice(results, func(i, j int) bool {
		return results[i].Slug < results[j].Slug
	})

	n.peersMu.Lock()
	n.peers = results
	n.peersMu.Unlock()
}

func (n *LinkNode) probePeerInfo(ctx context.Context, peerIP string) (*PeerInfoPayload, error) {
	url := fmt.Sprintf("http://%s:%d/link/v1/info", peerIP, n.cfg.MeshPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := n.dialClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}
	var payload PeerInfoPayload
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	return &payload, nil
}
