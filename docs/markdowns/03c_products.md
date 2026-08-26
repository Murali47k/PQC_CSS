
# What's Running in Production?

| Deployment               | Details                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Google Gboard**        | Since ~2017; tens of millions of phones; 600M+ example sentences; 1.5M users |
| **NVIDIA FLARE + Clara** | 20 hospitals built a COVID oxygen-needs model in 20 days                     |
| **Owkin + Substra**      | 10-pharma drug discovery partnership; multi-hospital cancer-tissue project   |
| **FATE**                 | Chinese banking; credit scoring across institutions                          |
| **Flower Labs**          | Local on-device AI and 2025 blockchain/DAO partnership                       |

### Medical-imaging benchmark

| Tool        | Strongest area                    |
| ----------- | --------------------------------- |
| **FLARE**   | Production scaling                |
| **Flower**  | Quick experimentation             |
| **Substra** | Privacy and regulatory compliance |



# Companies & Startups

## FL / Privacy-Tech Startups

| Company                  | Location          | Focus                                                         |
| ------------------------ | ----------------- | ------------------------------------------------------------- |
| **Flower Labs**          | Germany           | Federated AI across cloud, mobile and IoT; local on-device AI |
| **Apheris**              | Berlin            | Pharma and life sciences                                      |
| **Rhino Health**         | Boston + Tel Aviv | Hospital federated computing                                  |
| **Duality Technologies** | —                 | FL + cryptographic privacy                                    |
| **Sherpa.ai**            | Spain             | Clinical trials, financial crime                              |
| **Bitfount**             | London            | Life sciences, healthcare, finance                            |
| **Scaleout**             | Sweden            | Automotive, defense, cybersecurity                            |
| **Tune Insight**         | —                 | Encrypted computing, secure collaboration                     |
| **FLock.io**             | UK                | FL + blockchain + incentives                                  |
| **TripleBlind**          | Kansas City       | FL + multi-party computation                                  |

### Notable points

* **Apheris:** five pharmaceutical companies jointly trained a structural biology model in under ten weeks without sharing underlying data.
* **Rhino Health:** claims **50+ institutions** across the US, Israel, UK, Brazil and Asia-Pacific.
* **Duality Technologies:** combines FL with differential privacy, secret-sharing and encrypted parameters; worked with NHS England and the US National Cancer Institute.
* **Sherpa.ai:** combines FL with homomorphic encryption, secure multi-party computation and differential privacy.
* **Bitfount:** trained a retinal foundation model through its no-code platform.
* **FLock.io:** uses token staking and smart contracts to reward useful training updates.
* **TripleBlind:** acquired in 2024 and is no longer independent. 

---

## Larger Companies Adding FL

| Company       | Activity                                                    |
| ------------- | ----------------------------------------------------------- |
| **Microsoft** | Confidential Federated Learning in Azure ML using Intel SGX |
| **IBM**       | Federated fraud detection with 12 global banks              |
| **Cloudera**  | FL integrated into data governance                          |
| **Enveil**    | Encrypted FL through ZeroReveal                             |
| **Lifebit**   | Agentic federated genomics/healthcare platform              |


---

# Six Things to Watch

|     # | Point                                                                                |
| ----: | ------------------------------------------------------------------------------------ |
| **1** | No ready-made C++ FL coordinator                                                     |
| **2** | No agreed way to measure differences between participants' data                      |
| **3** | Secure aggregation is not a complete privacy guarantee                               |
| **4** | Federated LLM fine-tuning is still immature                                          |
| **5** | Major tools do not handle fairness/rewards by default                                |
| **6** | Older research often used artificially split data rather than separate organizations |
