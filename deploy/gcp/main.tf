# The public demo on Google Cloud: the API on Cloud Run with Redis beside it, Postgres on Cloud SQL, Gemini on
# Vertex AI, and a kill switch that takes the project off its billing account when the budget is spent or the
# last day comes. Everything lives in one project of its own, so deleting the project removes all of it.
#
#   Cloud Run "secop-analyst"   api (this repository's image) + redis (sidecar, memory only)
#      |  /cloudsql socket          |  Vertex AI, as the service's own account
#   Cloud SQL Postgres 17          Gemini
#
#   budget (gross cost, credits excluded) --+
#   Cloud Scheduler, on the last day -------+--> Pub/Sub "killswitch" --> function "killswitch" --> billing off

terraform {
  required_version = ">= 1.10"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.6"
    }
  }

  backend "gcs" {
    prefix = "secop-analyst/demo" # the bucket is given at init: terraform init -backend-config="bucket=<name>"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region

  default_labels = {
    project    = "secop-analyst"
    managed-by = "terraform"
  }
}

data "google_project" "this" {}

locals {
  services = [
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "billingbudgets.googleapis.com",
    "cloudbilling.googleapis.com",
    "cloudbuild.googleapis.com", # builds the function's container
    "cloudfunctions.googleapis.com",
    "cloudscheduler.googleapis.com",
    "eventarc.googleapis.com",
    "pubsub.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
  ]
}

resource "google_project_service" "this" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# The image is built and pushed by hand (see the README); Terraform only makes the place it goes.
resource "google_artifact_registry_repository" "images" {
  repository_id = "secop-analyst"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.this]
}

# ---- Postgres ----------------------------------------------------------------------------------------------

resource "google_sql_database_instance" "this" {
  name                = "secop-analyst"
  database_version    = "POSTGRES_17"
  region              = var.region
  deletion_protection = false # a demo: deleting the project is how it ends

  settings {
    edition           = "ENTERPRISE"  # the shared-core machines exist only in this edition
    tier              = "db-f1-micro" # shared core, the cheapest machine; the data is 36,000 rows
    availability_type = "ZONAL"
    disk_size         = 10

    # A public address with no authorised networks: nothing connects to it but the Cloud SQL connector, which
    # checks IAM and encrypts. Cloud Run uses the connector through its /cloudsql socket.
    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }

    backup_configuration {
      enabled = false # the contracts can be loaded again from SECOP; nothing else here is worth a backup
    }
  }

  depends_on = [google_project_service.this]
}

resource "google_sql_database" "secop" {
  name     = "secop"
  instance = google_sql_database_instance.this.name
}

resource "random_password" "postgres" {
  length  = 32
  special = false
}

# ponytail: the service connects as the built-in postgres user, which owns the imported tables. The SQL the
# model writes still runs as analyst_ro, which can read one table. Give the service a user of its own, with
# grants on what it writes, if anything beyond the demo runs on this instance.
resource "google_sql_user" "postgres" {
  name     = "postgres"
  instance = google_sql_database_instance.this.name
  password = random_password.postgres.result
}

resource "google_secret_manager_secret" "database_url" {
  secret_id = "database-url"
  replication {
    auto {}
  }
  depends_on = [google_project_service.this]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://postgres:${random_password.postgres.result}@/secop?host=/cloudsql/${google_sql_database_instance.this.connection_name}"
}

# ---- The service -------------------------------------------------------------------------------------------

resource "google_service_account" "api" {
  account_id   = "secop-analyst"
  display_name = "secop-analyst on Cloud Run"
}

resource "google_project_iam_member" "api" {
  for_each = toset(["roles/aiplatform.user", "roles/cloudsql.client"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "api" {
  secret_id = google_secret_manager_secret.database_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

locals {
  env = {
    MODEL_PROVIDER             = "vertex"
    GOOGLE_CLOUD_PROJECT       = var.project_id
    REDIS_URL                  = "redis://localhost:6379/0"
    DEMO_QUESTIONS_PER_DAY     = tostring(var.questions_per_day)
    DEMO_QUESTIONS_PER_VISITOR = tostring(var.questions_per_visitor)
  }
}

resource "google_cloud_run_v2_service" "api" {
  name                = "secop-analyst"
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.api.email

    scaling {
      min_instance_count = 1 # one warm instance, so the embedding model is already loaded
      max_instance_count = 2
    }

    containers {
      name  = "api"
      image = var.image
      ports {
        container_port = 8000
      }
      resources {
        limits   = { cpu = "1", memory = "2Gi" } # the embedding model is loaded in memory
        cpu_idle = true                          # CPU only during requests (streaming included), so the warm instance bills at the idle rate
      }
      dynamic "env" {
        for_each = local.env
        content {
          name  = env.key
          value = env.value
        }
      }
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
      startup_probe {
        http_get {
          path = "/readyz"
        }
        period_seconds    = 5
        failure_threshold = 24
      }
      depends_on = ["redis"]
    }

    # Redis holds the rate limits and the working memory. Beside the API it costs nothing extra and starts
    # empty with every instance; what is worth keeping is promoted to Postgres, and the daily question counts
    # are in Postgres. Memorystore would keep it across instances for about 36 USD a month.
    containers {
      name  = "redis"
      image = "docker.io/library/redis:8-alpine"
      resources {
        limits = { cpu = "1", memory = "512Mi" }
      }
      startup_probe {
        tcp_socket {
          port = 6379
        }
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.this.connection_name]
      }
    }
  }

  depends_on = [google_project_iam_member.api, google_secret_manager_secret_iam_member.api, google_secret_manager_secret_version.database_url]
}

# A public demo: anyone may call it. The application holds anonymous visitors to their daily questions.
resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Applies schema.sql to the database. Run it once after the import and after each schema change:
#   gcloud run jobs execute secop-analyst-migrate --region <region> --wait
resource "google_cloud_run_v2_job" "migrate" {
  name                = "secop-analyst-migrate"
  location            = var.region
  deletion_protection = false

  template {
    template {
      service_account = google_service_account.api.email
      max_retries     = 0
      containers {
        image   = var.image
        command = ["python", "db.py"]
        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.this.connection_name]
        }
      }
    }
  }

  depends_on = [google_project_iam_member.api, google_secret_manager_secret_iam_member.api, google_secret_manager_secret_version.database_url]
}

# ---- The kill switch ---------------------------------------------------------------------------------------

resource "google_pubsub_topic" "killswitch" {
  name       = "killswitch"
  depends_on = [google_project_service.this]
}

# Credits are left out of what the budget counts: with them in, a free-trial project spends nothing and the
# budget would never be reached.
resource "google_billing_budget" "demo" {
  billing_account = var.billing_account
  display_name    = "secop-analyst demo"

  budget_filter {
    projects               = ["projects/${data.google_project.this.number}"]
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.budget)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  all_updates_rule {
    pubsub_topic   = google_pubsub_topic.killswitch.id
    schema_version = "1.0"
  }

  depends_on = [google_project_service.this]
}

resource "google_cloud_scheduler_job" "deadline" {
  name      = "secop-analyst-deadline"
  region    = var.region
  schedule  = var.deadline_cron
  time_zone = "America/Bogota"

  pubsub_target {
    topic_name = google_pubsub_topic.killswitch.id
    data       = base64encode(jsonencode({ deadline = true }))
  }

  depends_on = [google_project_service.this]
}

resource "google_service_account" "killswitch" {
  account_id   = "killswitch"
  display_name = "Takes the project off its billing account"
}

# Enough to unlink the project from its billing account, and nothing on the billing account itself.
resource "google_project_iam_member" "killswitch" {
  project = var.project_id
  role    = "roles/billing.projectManager"
  member  = "serviceAccount:${google_service_account.killswitch.email}"
}

resource "google_storage_bucket" "functions" {
  name                        = "${var.project_id}-functions"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true
}

data "archive_file" "killswitch" {
  type        = "zip"
  source_dir  = "${path.module}/killswitch"
  output_path = "${path.module}/.build/killswitch.zip"
}

resource "google_storage_bucket_object" "killswitch" {
  name   = "killswitch-${data.archive_file.killswitch.output_md5}.zip"
  bucket = google_storage_bucket.functions.name
  source = data.archive_file.killswitch.output_path
}

resource "google_cloudfunctions2_function" "killswitch" {
  name     = "killswitch"
  location = var.region

  build_config {
    runtime     = "python313"
    entry_point = "stop"
    source {
      storage_source {
        bucket = google_storage_bucket.functions.name
        object = google_storage_bucket_object.killswitch.name
      }
    }
  }

  service_config {
    max_instance_count    = 1
    available_memory      = "256Mi"
    service_account_email = google_service_account.killswitch.email
    environment_variables = { PROJECT_ID = var.project_id }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.killswitch.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.killswitch.email
  }

  depends_on = [google_project_iam_member.killswitch, google_project_iam_member.killswitch_invoker]
}

# Eventarc delivers the Pub/Sub message to the function as this account.
resource "google_project_iam_member" "killswitch_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.killswitch.email}"
}
