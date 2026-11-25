use bibulm_core::{AppSettings, Database, default_data_dir};
use bibulm_hardware::HardwareMonitor;
use bibulm_link::LinkManager;
use bibulm_models::DownloadManager;
use bibulm_runtime::ModelSupervisor;
use bibulm_server::{ApiGateway, GatewayContext, LinkHardwareSnapshot};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

pub struct AppState {
    pub db: Database,
    pub settings: Arc<RwLock<AppSettings>>,
    pub hardware: Arc<Mutex<HardwareMonitor>>,
    pub supervisor: Arc<ModelSupervisor>,
    pub downloads: Arc<DownloadManager>,
    pub link: Arc<LinkManager>,
    pub link_hw_snapshot: Arc<RwLock<LinkHardwareSnapshot>>,
    pub gateway: Arc<ApiGateway>,
    pub allow_exit: Arc<AtomicBool>,
}

impl AppState {
    pub fn initialize() -> anyhow::Result<Self> {
        let data_dir = default_data_dir();
        std::fs::create_dir_all(&data_dir)?;
        let db_path = data_dir.join("bibulm.db");

        let db = Database::open(&db_path)?;

        let settings_val = db.load_settings().unwrap_or_default();
        // Ensure configured directories exist
        let _ = std::fs::create_dir_all(settings_val.primary_models_dir());
        let _ = std::fs::create_dir_all(&settings_val.runtimes_dir);
        let _ = std::fs::create_dir_all(&settings_val.logs_dir);

        let initial_link_cfg = settings_val.link.clone();
        let settings = Arc::new(RwLock::new(settings_val));
        let mut hw_monitor = HardwareMonitor::new();
        let link_hw_snapshot = Arc::new(RwLock::new(LinkHardwareSnapshot::from_monitor(
            &mut hw_monitor,
            &[],
        )));

        let hardware = Arc::new(Mutex::new(hw_monitor));
        let supervisor = Arc::new(ModelSupervisor::new());
        let downloads = Arc::new(DownloadManager::new(db.clone(), 2));
        let link = Arc::new(LinkManager::new(&initial_link_cfg));
        let gw_ctx = GatewayContext::new(db.clone(), supervisor.clone(), settings.clone())
            .with_link(link.clone())
            .with_hw_snapshot(link_hw_snapshot.clone())
            .with_downloads(downloads.clone());
        let gateway = Arc::new(ApiGateway::new(gw_ctx));

        Ok(Self {
            db,
            settings,
            hardware,
            supervisor,
            downloads,
            link,
            link_hw_snapshot,
            gateway,
            allow_exit: Arc::new(AtomicBool::new(false)),
        })
    }
}
