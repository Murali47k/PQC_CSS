

# Solved vs Open Problems

| Problem                                 | Status | Key point                                                                      |
| --------------------------------------- | :----: | ------------------------------------------------------------------------------ |
| Basic training                          |    ✅   | FedAvg converges reliably with reasonably similar data and enough participants |
| Communication compression               |    ✅   | Updates can be compressed to save bandwidth                                    |
| Secure aggregation                      |    ✅   | Server can calculate the sum without seeing individual updates                 |
| Basic poisoning resistance              |    ✅   | Simple sabotage is less effective than expected                                |
| **Non-IID data**                        |    ❌   | No agreed way to measure how different data distributions are                  |
| **Privacy leaks**                       |    ❌   | Aggregate updates can still reveal underlying data                             |
| **Malicious participants + messy data** |    ❌   | Difficult to distinguish unusual data from attacks                             |
| **Federated LLM fine-tuning**           |    ❌   | Several unresolved issues                                                      |
| **Device unreliability**                |    ❌   | Devices drop out during rounds                                                 |
| **Fairness/rewards**                    |    ❌   | No major tool provides this by default                                         |



## Non-IID Data

Different participants have different data distributions.

```text
Hospital A → Patient data A
Hospital B → Patient data B
Hospital C → Patient data C
```

There is still no agreed way to **measure how different** these distributions are. Testing on artificially uniform data can therefore give misleading results. 

## Privacy

Secure aggregation hides individual updates but still exposes the **average update**.

Collecting enough averages over time can allow reconstruction of underlying data.

**Current approach:** add statistical noise → **local differential privacy**

**Trade-off:** more privacy ↔ lower model accuracy. 

## Malicious Participants

When data is already very different:

> “This update is unusual because the data is different”
> vs.
> “This update is unusual because someone is attacking”

No current method handles **messy data + unreliable devices + compressed communication** simultaneously without trade-offs. 


## Federated LLM Fine-Tuning

| Issue                       | Problem                                                      |
| --------------------------- | ------------------------------------------------------------ |
| **Model size**              | Full model updates are too large                             |
| **LoRA**                    | Trains small adapter pieces instead                          |
| **Adapter aggregation**     | Some implementations combine adapters incorrectly            |
| **Privacy noise**           | Adds further instability                                     |
| **Different devices**       | Different memory/compute can require different adapter sizes |
| **Different adapter sizes** | Combining them remains unsolved                              |

The adapter aggregation correctness issue was still present in shared code as of early 2026. 

---

## Real-World Reliability

Google's production system found:

| Observation                   |       Result |
| ----------------------------- | -----------: |
| Phones dropping out mid-round |    **6–10%** |
| Additional devices required   | **30% more** |

Most published research still assumes every device stays connected and cooperative throughout the round. 

---

## Fairness & Rewards

Major FL tools do not have built-in mechanisms to:

* Identify repeatedly low-quality or harmful participants
* Reduce their trust
* Reduce their compensation
* Reward useful participants

This remains a niche research problem and a business problem as well. 

---
