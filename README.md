# LoanEligibilityPrediction
A machine learning model that predicts loan default risk using logistic regression.
# 🧠 Loan Eligibility Prediction

This project predicts whether a loan applicant is likely to **fully repay** a loan or **default** based on historical financial data. It uses logistic regression as a binary classification model.

---

## 📌 Problem Statement

Given a set of loan applicant features (like credit history, income, loan purpose, etc.), build a model to predict whether the applicant will default on a loan.

---

## 📂 Dataset

- File: `loan_data.csv`
- Target column: `not.fully.paid` (1 = will default, 0 = will repay)
- Features include:
  - `credit.policy`
  - `purpose` (categorical)
  - `int.rate`
  - `installment`
  - `log.annual.inc`
  - `dti`
  - `revol.bal`
  - `revol.util`
  - `inq.last.6mths`
  - `delinq.2yrs`
  - `pub.rec`

---

## 🛠️ Tech Stack

| Tool / Library     | Purpose                        |
|--------------------|--------------------------------|
| Python             | Programming language           |
| Pandas / NumPy     | Data manipulation              |
| Scikit-learn       | ML modeling and evaluation     |
| Seaborn / Matplotlib | Data visualization          |
| Logistic Regression | Classification algorithm      |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-eligibility-prediction.git
   cd loan-eligibility-prediction

