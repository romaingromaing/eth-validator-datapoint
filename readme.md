# Ethereum Validator Analysis Platform
## Enterprise Data Engineering Pipeline & Analytics Dashboard

A production-grade data engineering platform for comprehensive Ethereum validator analysis. This system implements automated ETL pipelines, multi-API data enrichment, secure role-based access control, and scheduled batch processing with an interactive analytics dashboard.

## Architecture Overview

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        BC[BeaconChain API<br/>Validator Metadata]
        DA[Dune Analytics Sim API<br/>Transaction Analysis]
        DEX[Dune Analytics Client<br/>DEX Address Registry]
    end
    
    subgraph "Data Processing Engine"
        PS[Python ETL Pipeline<br/>validator_analysis.py]
        SCH[APScheduler<br/>Automated Jobs]
    end
    
    subgraph "Data Warehouse"
        PG[(Supabase PostgreSQL<br/>Production Database)]
        CSV[CSV Export Layer<br/>Data Lake Output]
    end
    
    subgraph "Analytics & Access Layer"
        ADMIN[Admin Panel<br/>Full Access Dashboard]
        READ[Read-Only Dashboard<br/>Public Analytics]
        AUTH[Access Control System<br/>Role-Based Security]
    end
    
    BC --> PS
    DA --> PS
    DEX --> PS
    SCH --> PS
    PS --> PG
    PS --> CSV
    PG --> ADMIN
    PG --> READ
    AUTH --> ADMIN
    AUTH --> READ
    
    classDef ingestion fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef frontend fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    
    class BC,DA,DEX ingestion
    class PS,SCH processing
    class PG,CSV storage
    class ADMIN,READ,AUTH frontend
```

## Key Features

### Data Engineering Pipeline
- **Multi-Source ETL**: Orchestrates data ingestion from BeaconChain API, Dune Analytics, and custom DEX registries
- **Smart Contract Detection**: Advanced transaction analysis to identify contract deployment activities
- **Batch Processing Architecture**: Configurable batch sizes with intelligent rate limiting and error recovery
- **Data Quality Assurance**: Comprehensive validation, deduplication, and enrichment processes
- **Automated Scheduling**: Production-ready cron jobs with APScheduler for monthly data refreshes

### Enterprise Security & Access Control
- **Role-Based Access Control (RBAC)**: Secure admin authentication with password-based access
- **Admin Panel**: Full-featured management interface with data refresh, analysis triggers, and system monitoring
- **Read-Only Dashboard**: Public analytics interface for stakeholder access without system privileges
- **Environment-Based Security**: Secure credential management with multi-tier configuration support

### Analytics Dashboard
- **Interactive Filtering**: Multi-dimensional data slicing by validator status, smart contract activity, and DEX classification
- **Real-Time Metrics**: Live dashboard with validator status tracking and deposit address analytics
- **Data Visualization**: Distribution analysis, timeline charts, and comprehensive KPI monitoring
- **Export Capabilities**: Filtered CSV exports with timestamp-based file naming

### Automated Operations
- **Scheduled Processing**: Monthly automated data pipeline execution via integrated cron scheduler
- **Manual Triggers**: On-demand analysis execution with real-time progress monitoring
- **Run History**: Comprehensive logging and status tracking for all automated and manual executions
- **Error Handling**: Robust exception handling with timeout protection and failure notifications

## Data Engineering Architecture

### ETL Pipeline Components

1. **Extract Phase**
   - BeaconChain API integration for validator metadata
   - Dune Analytics Sim API for transaction history analysis  
   - Dune Analytics Client for DEX address registry synchronization

2. **Transform Phase**
   - Deposit address enrichment and validation
   - Smart contract deployment detection algorithms
   - DEX address classification and cross-referencing
   - Data normalization and quality checks

3. **Load Phase**
   - PostgreSQL database upserts with conflict resolution
   - CSV data lake exports for downstream analysis
   - Real-time dashboard cache invalidation

### Scheduler Integration

```python
# Automated monthly execution
ENABLE_AUTO_ANALYSIS=true
CRON_DAY=1          # First day of month
CRON_HOUR=2         # 2 AM UTC execution
CRON_MINUTE=0       # Top of the hour
```

## Installation & Configuration

### Prerequisites
- Python 3.8+
- Supabase PostgreSQL instance
- Dune Analytics API credentials
- Production environment variables

### Environment Setup

```bash
# Clone repository
git clone https://github.com/0xhaisenberg/eth-validator-analysis.git
cd eth-validator-analysis

# Install dependencies
pip install -r requirements.txt
```

### Production Configuration

Create `.env` file with production credentials:
```env
# API Credentials
DUNE_SIM_API_KEY=your_dune_sim_key
DUNE_CLIENT_API_KEY=your_dune_client_key

# Database Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_DATABASE_URL=postgresql://user:pass@host:5432/db
SUPABASE_TABLE_NAME=validator_data

# Access Control
ADMIN_PASSWORD=your_secure_admin_password
RESTRICT_ADMIN_ACCESS=true

# Automation Settings
ENABLE_AUTO_ANALYSIS=true
CRON_DAY=1
CRON_HOUR=2
CRON_MINUTE=0

# Performance Tuning
BATCH_SIZE=100
DELAY_SECONDS=15
API_DELAY=0.25
```

### Database Schema

```sql
-- Production database schema
CREATE TABLE validator_data (
    id BIGSERIAL PRIMARY KEY,
    index INTEGER NOT NULL,
    pubkey TEXT NOT NULL,
    deposit_address TEXT,
    last_transaction_time TIMESTAMPTZ,
    is_smart_contract BOOLEAN DEFAULT FALSE,
    is_dex BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_validator_deposit ON validator_data(deposit_address);
CREATE INDEX idx_validator_contract ON validator_data(is_smart_contract);
CREATE INDEX idx_validator_dex ON validator_data(is_dex);
CREATE INDEX idx_validator_activity ON validator_data(last_transaction_time);
CREATE INDEX idx_validator_created ON validator_data(created_at);
```

## Production Deployment

### Manual Execution
```bash
# Run ETL pipeline
python validator_analysis.py

# Launch dashboard
streamlit run streamlit_app.py
```

### Automated Scheduling
The platform includes built-in scheduling capabilities:
- Monthly automated data pipeline execution
- Configurable execution timing (day/hour/minute)
- Manual trigger capabilities for ad-hoc analysis
- Comprehensive run history and status monitoring

## Access Control System

### Admin Panel Features
- Full data refresh capabilities
- Manual analysis trigger with real-time progress monitoring
- Scheduler management (start/stop/configure)
- System status monitoring and run history
- Complete database access and export functionality

### Read-Only Dashboard
- Comprehensive validator analytics without system access
- Interactive filtering and visualization
- Data export capabilities
- Real-time metrics and KPI monitoring
- No administrative functions or data modification capabilities

## Data Schema & Analytics

### Core Metrics Tracked
- **Validator Status**: Active/inactive validator distribution
- **Deposit Address Analysis**: Unique addresses with source classification
- **Smart Contract Activity**: Contract deployment identification and tracking  
- **DEX Integration**: Known DEX address classification and analysis
- **Transaction Timeline**: Historical activity patterns and trends

### Analytics Capabilities
- Multi-dimensional filtering across all validator attributes
- Time-series analysis of validator activity patterns
- Distribution analysis of deposit sources (wallet/contract/DEX)
- Export functionality for downstream analysis and reporting

## API Integration Architecture

### BeaconChain API Integration
- Validator metadata and deposit address enrichment
- Batch processing with intelligent rate limiting
- Error recovery and retry logic

### Dune Analytics Integration
- **Sim API**: Transaction history analysis and smart contract detection
- **Client API**: DEX address registry synchronization
- Configurable query execution with timeout handling

## Security & Compliance

### Access Control
- Environment-based credential management
- Role-based dashboard access with secure authentication
- Admin panel protection with configurable access restrictions

### Data Protection
- Secure API key management
- Database connection encryption
- No sensitive data exposure in logs or exports

## Monitoring & Operations

### Scheduler Monitoring
- Real-time job status tracking
- Historical run analysis with success/failure rates
- Manual intervention capabilities
- Comprehensive error logging and alerting

### Performance Metrics
- Batch processing performance monitoring
- API rate limit compliance tracking
- Database operation performance analysis
- Dashboard response time optimization

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run data pipeline
python validator_analysis.py

# Launch development dashboard
streamlit run streamlit_app.py
```

### Code Quality Standards
- Comprehensive error handling and logging
- Type hints and documentation
- Environment-based configuration management
- Secure credential handling

## Production Considerations

### Scalability
- Configurable batch processing for large datasets
- Database connection pooling and optimization
- Efficient API rate limit management
- Streamlined data pipeline with minimal resource usage

### Reliability
- Comprehensive error handling with graceful degradation
- Automated retry logic for transient failures
- Data consistency checks and validation
- Robust scheduler with failure recovery

### Monitoring
- Detailed logging for all pipeline operations
- Performance metrics collection and analysis
- Alert system for critical failures
- Comprehensive audit trail for all administrative actions

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BeaconChain API](https://beaconcha.in/api/v1/docs) for validator data
- [Dune Analytics](https://dune.com) for transaction and DEX data
- [Supabase](https://supabase.com) for database services
- [Streamlit](https://streamlit.io) for the dashboard framework

## Support

For technical issues, feature requests, or deployment assistance, please create an issue in the GitHub repository with detailed system information and error logs.

⭐ **Star this repo** if you find it useful!
