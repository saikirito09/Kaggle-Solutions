# Home Credit Default Risk Prediction

## Objective

The objective of this competition is to predict which clients are more likely to default on their loans. The goal is to develop models that are not only accurate but also stable over time, ensuring that the predictions remain reliable in the future. This will help consumer finance providers to better assess potential clients' default risks, enabling them to offer loans to a broader population, including those with little or no credit history.

### Dataset Description

The datasets are organized into three depths, each representing different levels of historical records associated with a specific `CASE_ID`.

#### Depth 0

- **Static Features**: These features are directly tied to a specific `CASE_ID`.
  - **Internal Files**: `static_0.csv`
  - **External Files**: `static_cb_0.csv`
- **Training Files**:
  - Internal: `train_static_0.0.csv`, `train_static_0.1.csv`
  - External: `train_static_cb_0.csv`

#### Depth 1

- **Historical Records Indexed by `num_group1`**: Each `CASE_ID` has associated records indexed by `num_group1`.
  - **Internal Source**: `applprev_1`, `other_1`, `deposit_1`, `person_1`, `debtload1`
  - **External Source**: `tax_registry_a1`, `tax_registry_b1`, `credit_bureau_a1`, `credit_bureau_b1`
- **Training Files**:
  - Internal: `train_applprev_1.0.csv`, `train_applprev_1.1.csv`, `train_other_1.csv`, `train_deposit_1.csv`, `train_person_1.csv`, `train_debtload1.csv`
  - External: `train_tax_registry_a1.csv`, `train_tax_registry_b1.csv`, `train_credit_bureau_a1.0.csv`, `train_credit_bureau_a1.1.csv`, `train_credit_bureau_b1.0.csv`, `train_credit_bureau_b1.1.csv`

#### Depth 2

- **Historical Records Indexed by Both `num_group1` and `num_group2`**: Each `CASE_ID` has associated records indexed by both groups.
  - **Internal Source**: `applprev_2`, `person_2`
  - **External Source**: `credit_bureau_a2`, `credit_bureau_b2`
- **Training Files**:
  - Internal: `train_applprev_2.csv`, `train_person_2.csv`
  - External: `train_credit_bureau_a2.0.csv`, `train_credit_bureau_a2.1.csv`, `train_credit_bureau_a2.2.csv`, `train_credit_bureau_b2.0.csv`, `train_credit_bureau_b2.1.csv`, `train_credit_bureau_b2.2.csv`

### Feature Groups and Column Examples

- **Feature Groups**:
  - P: Transform DPD (Days Past Due)
  - M: Masking Categories
  - A: Transform Amount
  - D: Transform Date
  - T: Unspecified Transform
  - L: Unspecified Transform
- **Columns Examples**:
  - P: `actualdpd_943P`
  - M: `maritalstat_385M`
  - A: `pmtssum_45A`
  - D: `dateofbirth_337D`
  - T: `riskassessment_940T`
  - L: `pmtcount_4955617L`

### Objective

The key objective is to develop a model that not only predicts loan defaults accurately but also remains stable over time, ensuring consistent performance as client behaviors change.

### Impact

Your work in this project will help consumer finance providers improve their methods for assessing potential clients' default risks, making loans more accessible to individuals with little or no credit history, thereby promoting financial inclusion.

### Usage

1. **Data Preparation**: Organize and preprocess the data from different depths.
2. **Exploratory Data Analysis (EDA)**: Analyze the data to identify patterns and relationships.
3. **Model Development**: Develop and train models to predict loan defaults.
4. **Model Evaluation**: Ensure the models are stable and perform well over time.

### Contributions

We welcome contributions from the community to improve the project. Please follow the guidelines for contributing and ensure all changes are well-documented.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.
