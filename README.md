# [Project Title] - Problem Framing  
**Stage:** 01 (Problem Framing & Scoping)  

## Problem Statement  
Bank needs to predict small business loan defaults" because business impact.

## Stakeholder & User  
- **Decision Owner:** Chief Risk Officer
- **Tool User:** Underwriting Team"
- **Decision Window:** Real-time during loan applications and monthly model retraining

## Useful Answer & Decision 
- **Predictive:** Output: Default probability score (0-100%)"

## Assumptions & Constraints
- Data: 3 years of historical loan data available
- Constraints: Model latency <500ms for real-time use

## Known Unknowns / Risks 
- Risk: Regulatory changes impacting model features

## Lifecycle Plan  
- build predictive model → modeling (Stage 01) → python module in /src/ with report

## Repo plan
/data/, /src/, /notebooks/, /docs/ ; cadence for updates