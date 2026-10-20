output "url" {
  description = "The demo's address."
  value       = google_cloud_run_v2_service.api.uri
}

output "image_repository" {
  description = "Where to push the image: docker push <this>/api:<tag>"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "sql_instance" {
  description = "The Cloud SQL instance, for gcloud sql import."
  value       = google_sql_database_instance.this.name
}
