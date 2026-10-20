variable "project_id" {
  description = "The demo's own project. The kill switch takes this whole project off its billing account."
  type        = string
}

variable "region" {
  description = "Region of every resource."
  type        = string
  default     = "us-central1"
}

variable "image" {
  description = "The API image, pushed to the Artifact Registry repository this creates, by digest."
  type        = string
}

variable "billing_account" {
  description = "Billing account the project is linked to, for the budget."
  type        = string
}

variable "budget" {
  description = "Gross spend, credits left out, at which the kill switch takes the project off its billing account."
  type        = number

  validation {
    condition     = var.budget > 0
    error_message = "budget is a positive amount."
  }
}

variable "budget_currency" {
  description = "Currency of the billing account, such as USD or COP. A budget in another currency is rejected."
  type        = string
  default     = "USD"
}

variable "deadline_cron" {
  description = "When the kill switch runs regardless of spend, in cron syntax, Bogotá time. Before the credits expire."
  type        = string
}

variable "questions_per_day" {
  description = "Questions all anonymous visitors together may ask in a day."
  type        = number
  default     = 60
}

variable "questions_per_visitor" {
  description = "Questions one anonymous visitor may ask in a day."
  type        = number
  default     = 10
}
