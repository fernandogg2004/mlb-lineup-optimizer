# terraform/main.tf — Infraestructura AWS para MLB AI
# Provisiona: S3 bucket, IAM roles, RDS, ECR, ECS Cluster
# Uso: terraform init && terraform plan && terraform apply

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "mlb-ai-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "mlb-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

variable "aws_region"   { default = "us-east-1" }
variable "environment"  { default = "production" }
variable "db_password"  { sensitive = true }

# ── S3 Datalake ──────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "datalake" {
  bucket = "mlb-ai-datalake"
}

resource "aws_s3_bucket_versioning" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket                  = aws_s3_bucket.datalake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── IAM Role para ECS Task de Inferencia ────────────────────────────────────
resource "aws_iam_role" "inference_task" {
  name = "mlb-ai-inference-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_policy" "inference_policy" {
  name        = "mlb-ai-inference-policy"
  description = "Mínimo privilegio para el servicio de inferencia MLB AI"
  policy      = file("${path.module}/../inference_role_policy.json")
}

resource "aws_iam_role_policy_attachment" "inference" {
  role       = aws_iam_role.inference_task.name
  policy_arn = aws_iam_policy.inference_policy.arn
}

# ── ECR Repository para la imagen Docker ────────────────────────────────────
resource "aws_ecr_repository" "api" {
  name                 = "mlb-ai-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "KMS"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Mantener solo las 10 imágenes más recientes"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── RDS PostgreSQL ───────────────────────────────────────────────────────────
resource "aws_db_instance" "predictions" {
  identifier              = "mlb-ai-predictions"
  engine                  = "postgres"
  engine_version          = "16.2"
  instance_class          = "db.t3.small"
  allocated_storage       = 20
  max_allocated_storage   = 100
  storage_encrypted       = true
  db_name                 = "mlb_predictions"
  username                = "mlb"
  password                = var.db_password
  skip_final_snapshot     = false
  final_snapshot_identifier = "mlb-ai-predictions-final"
  backup_retention_period = 7
  deletion_protection     = true

  tags = {
    Name = "mlb-ai-predictions-db"
  }
}

# ── Secrets Manager para API Token ──────────────────────────────────────────
resource "aws_secretsmanager_secret" "api_token" {
  name                    = "mlb-ai/api-token"
  recovery_window_in_days = 7
}

# ── Outputs ──────────────────────────────────────────────────────────────────
output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "rds_endpoint" {
  value     = aws_db_instance.predictions.endpoint
  sensitive = true
}

output "inference_role_arn" {
  value = aws_iam_role.inference_task.arn
}
