# Credit Risk Prediction

This accelerator demos acceleration on GPUs for structured data by leveraging Nvidia Rapids. While we use Fannie Mae's [single family loan performance dataset](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data) to predict a customer is going to be delinquent on the mortgage payments, this can be easily adopted for any huge structured datasets. Most of the preprocessing and model building tasks done by NumPy, Pandas and Scikit-Learn are supported.
* Predict if a customer will be delinquent on credit payment.
* Uses gpu acceleration to significantly reduce running time on huge structured data.
* [SHAP](#https://github.com/slundberg/shap) to explain the predictions.

## Getting Started

To get started with this project, follow these steps:
1. Clone the repository to Azure ML workspace or your local machine.
2. Create the environment. Dockerfile can be found [here](configuration/environment/docker/Dockerfile).
3. Download the data by running [download_data](download_data.ipynb).
4. [Train](cudf_credit_risk.ipynb) the model leveraging gpu acceleration.

## Prerequisites

- Access to an Azure subscription
- Access to an Azure ML workspace

## Deploy Infrastructure

A guide to deploy the infrastructure for this project can be found [here](.azureDevOps/Deploy-Infrastructure.md).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
