variable "basename" {
  type        = string
  description = "Basename of the module."
  validation {
    condition     = can(regex("^[-\\w]{0,27}$", var.basename))
    error_message = "The name must be between 0 and 27 characters, can contain only letters, numbers, hyphens."
  }
  default = "gpu-cluster"
}

variable "location" {
  type        = string
  description = "Location of the resource group."
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "A mapping of tags which should be assigned to the deployed resource."
}

variable "machine_learning_workspace_id" {
  type        = string
  description = "The ID of the Machine Learning workspace."
}


variable "vm_priority" {
  type        = string
  description = "The priority of the VM."
  validation {
    condition     = contains(["dedicate", "lowpriority"], lower(var.vm_priority))
    error_message = "Valid values for vm_priority are \"Dedicated\", or \"LowPriority\"."
  }
  default = "LowPriority"
}

variable "vm_size" {
  type        = string
  description = "The size of the VM."
  default     = "Standard_NC6s_v3"
}
