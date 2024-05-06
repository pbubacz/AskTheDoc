# AskTheDoc

The application enables you to pose questions about documents. Currently, it supports only DOCX and PDF file formats.

## Prerequisites
- **Azure OpenAI Deployment**: To obtain the necessary key and endpoint, refer to [Retrieve Key and Endpoint](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python#retrieve-key-and-endpoint). For comprehensive application functionality, deploy the following models: GPT-3.5-Turbo, GPT-4, and GPT-4-Turbo. For more details on model availability, visit [Azure OpenAI Service Models Availability](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models).

- **Azure Document Intelligence Deployment**: Acquire the Endpoint URL and keys by following the instructions [here](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0#get-endpoint-url-and-keys).

## Deployment
Click the button below to deploy the application to Azure.  

[![Deploy To Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fpbubacz%2FAskTheDoc%2Fmain%2Fazuredeploy.json)