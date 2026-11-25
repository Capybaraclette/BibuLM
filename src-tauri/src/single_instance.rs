use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
struct InstanceEndpoint {
    port: u16,
    token: String,
}

/// Cross-platform single-instance lock backed by an OS kernel exclusive file lock
/// (`File::try_lock`). If another instance is already running, the new process signals
/// the primary instance to restore & focus its window and exits immediately.
pub struct SingleInstanceGuard {
    _lock_file: File,
    listener: Option<TcpListener>,
    token: String,
    endpoint_path: PathBuf,
    wakeup_path: PathBuf,
    stop_flag: Arc<AtomicBool>,
}

impl SingleInstanceGuard {
    /// Attempts to acquire the exclusive application instance lock inside `data_dir`.
    ///
    /// - Returns `Some(guard)` if this process is the primary instance.
    /// - Returns `None` if another instance is already running (after signaling it to focus).
    pub fn acquire(data_dir: &Path) -> Option<Self> {
        let _ = std::fs::create_dir_all(data_dir);
        let lock_path = data_dir.join("instance.lock");
        let endpoint_path = data_dir.join("instance.endpoint.json");
        let wakeup_path = data_dir.join("instance.wakeup");

        let lock_file = match OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("Could not open single-instance lock file: {e}");
                return None;
            }
        };

        match lock_file.try_lock() {
            Ok(()) => {
                let _ = std::fs::remove_file(&wakeup_path);
                let listener = TcpListener::bind("127.0.0.1:0").ok();
                let token = Uuid::new_v4().to_string();
                let port = listener
                    .as_ref()
                    .and_then(|ln| ln.local_addr().ok())
                    .map(|a| a.port())
                    .unwrap_or(0);
                let ep = InstanceEndpoint {
                    port,
                    token: token.clone(),
                };
                if let Ok(json) = serde_json::to_string(&ep) {
                    let _ = std::fs::write(&endpoint_path, json);
                }
                Some(Self {
                    _lock_file: lock_file,
                    listener,
                    token,
                    endpoint_path,
                    wakeup_path,
                    stop_flag: Arc::new(AtomicBool::new(false)),
                })
            }
            Err(TryLockError::WouldBlock) => {
                notify_existing_instance(&endpoint_path, &wakeup_path);
                None
            }
            Err(TryLockError::Error(e)) => {
                tracing::warn!("Single-instance lock error: {e}");
                notify_existing_instance(&endpoint_path, &wakeup_path);
                None
            }
        }
    }

    /// Spawns a lightweight background thread that invokes `on_focus` whenever a secondary
    /// instance attempts to launch and passes the matching wakeup token.
    pub fn spawn_wakeup_listener<F>(&mut self, on_focus: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        let listener = self.listener.take();
        if let Some(ref ln) = listener {
            let _ = ln.set_nonblocking(true);
        }
        let expected_token = self.token.clone();
        let wakeup_path = self.wakeup_path.clone();
        let stop_flag = self.stop_flag.clone();

        std::thread::Builder::new()
            .name("bibulm-single-instance".to_string())
            .spawn(move || {
                while !stop_flag.load(Ordering::Relaxed) {
                    let mut woke = false;

                    // Fast path: loopback TCP signal
                    if let Some(ref ln) = listener {
                        while let Ok((stream, _)) = ln.accept() {
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
                            let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
                            let mut reader = BufReader::new(stream);
                            let mut line = String::new();
                            if reader.read_line(&mut line).is_ok()
                                && line.trim() == expected_token
                            {
                                on_focus();
                                woke = true;
                                let mut writer = reader.into_inner();
                                let _ = writeln!(writer, "ok");
                                let _ = writer.flush();
                            }
                        }
                    }

                    // Fallback path: wakeup marker file (handles strict firewalls blocking loopback)
                    if wakeup_path.exists() {
                        if let Ok(content) = std::fs::read_to_string(&wakeup_path) {
                            let _ = std::fs::remove_file(&wakeup_path);
                            if content.trim() == expected_token && !woke {
                                on_focus();
                            }
                        } else {
                            let _ = std::fs::remove_file(&wakeup_path);
                        }
                    }

                    std::thread::sleep(Duration::from_millis(80));
                }
            })
            .ok();
    }
}

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        let _ = std::fs::remove_file(&self.endpoint_path);
        let _ = std::fs::remove_file(&self.wakeup_path);
    }
}

fn notify_existing_instance(endpoint_path: &Path, wakeup_path: &Path) -> bool {
    // Brief retry window in case the primary instance is mid-startup writing the endpoint file.
    for _ in 0..5 {
        if let Ok(raw) = std::fs::read_to_string(endpoint_path)
            && let Ok(ep) = serde_json::from_str::<InstanceEndpoint>(&raw)
        {
            if ep.port > 0 {
                let addr = SocketAddr::from(([127, 0, 0, 1], ep.port));
                if let Ok(mut stream) =
                    TcpStream::connect_timeout(&addr, Duration::from_millis(300))
                {
                    let _ = stream.set_read_timeout(Some(Duration::from_millis(300)));
                    let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));
                    let _ = writeln!(stream, "{}", ep.token);
                    let _ = stream.flush();
                    let mut ack = String::new();
                    let mut reader = BufReader::new(stream);
                    let _ = reader.read_line(&mut ack);
                    return true;
                }
            }
            // Fallback: write the verified token to the wakeup file
            if std::fs::write(wakeup_path, &ep.token).is_ok() {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn second_instance_is_blocked_and_wakes_primary() {
        let dir = std::env::temp_dir().join(format!("bibulm-si-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        let mut primary = SingleInstanceGuard::acquire(&dir).expect("primary should acquire lock");
        let wakeups = Arc::new(AtomicUsize::new(0));
        let w_clone = wakeups.clone();
        primary.spawn_wakeup_listener(move || {
            w_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Second attempt must fail to acquire and signal the primary
        assert!(SingleInstanceGuard::acquire(&dir).is_none());

        // Wait briefly for the listener thread to process the wakeup
        for _ in 0..20 {
            if wakeups.load(Ordering::SeqCst) >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(30));
        }
        assert_eq!(wakeups.load(Ordering::SeqCst), 1);

        // Once primary drops, a new instance can acquire the lock cleanly
        drop(primary);
        let next = SingleInstanceGuard::acquire(&dir);
        assert!(next.is_some());
        drop(next);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
