# Federated Learning

> **Federated Learning (FL)** trains a machine learning model across many devices or organizations **without moving raw data to one place**. Each device trains on its own data and sends only model updates to a central server which combines them. 

---

Almost everything usable today is built in **Python**.

| Tool                                                                    | Main use                   | Strengths                                                                              | Limitations                                                                        |
| ----------------------------------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Flower**                                                              | General-purpose FL         | PyTorch, TensorFlow, JAX, scikit-learn; Docker/Kubernetes; access controls; monitoring | Secure aggregation/privacy are add-ons; non-Python clients require manual plumbing |
| **NVIDIA FLARE**                                                        | Production                 | Security, identity, administration; hospitals and banks                                | PKI and admin setup take effort                                                    |
| **OpenFL**                                                              | Healthcare                 | PyTorch, TensorFlow; Linux Foundation                                                  | Feels dated compared with Flower/FLARE, especially for LLMs                        |
| **FATE**                                                                | Chinese financial services | Vertical FL; different institutions can hold different customer information            | FedAvg only out of the box; Java + Python; difficult outside its ecosystem         |
| **PySyft**                                                              | Privacy research           | Secure multi-party computation and other privacy techniques                            | More research-oriented than a ready-to-use FL system                               |
| **TensorFlow Federated**                                                | TensorFlow                 | Good for teams already using TensorFlow                                                | Google's investment appears to have slowed                                         |
| **FedML, FederatedScope, Fed-BioMed, IBM FL, FLUTE, HP Swarm Learning** | Niche use cases            | Benchmarking, async training, biomedical compliance, etc.                              | Less momentum than Flower/FLARE                                                    |

A 2024 comparison of 15 open-source FL tools scored **Flower highest by a wide margin**. 

### C++

There is **no mature general-purpose C++ FL orchestrator**.

| Part            | Approach                      |
| --------------- | ----------------------------- |
| Server          | Python + Flower / FLARE       |
| Client training | C++ + libtorch                |
| Communication   | Flower's existing protocol    |
| Missing piece   | C++ ↔ Python connective layer |

PULP-TrainLib and EdgeRL focus on training on tiny embedded chips rather than coordinating a whole FL system. 

---