class ConfigService:
    # hard code config for now
    database_url: str = "sqlite:///app.db"
    framestep: int = 1
    cv_threshold_tracking: float = 0.6
    cv_threshold_detection: float = 0.3
    cv_time_min_tracking: float = 0.5
