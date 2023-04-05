variable "name" {
    type    = string
    default = "credit-risk"
}

variable "location" {
    type    = string
    default = "West Europe"  
}

variable "postfix" {
    type    = string
    default = "dev"
}

resource "random_string" "postfix" {
  length  = 3
  special = false
  upper   = false
}