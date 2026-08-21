# 📈 Product & Growth Intelligence Platform

An interactive product analytics and decision-support application built to investigate user behavior, acquisition, engagement, funnel performance, retention, and growth opportunities.

🔗 **Live Application:** https://kaustubhi-growth-intelligence.streamlit.app/

---

## 🚀 Project Overview

Product teams generate large volumes of event-level data, but raw data alone does not provide actionable insights.

The **Product & Growth Intelligence Platform** transforms user activity data into an interactive analytics application where users can explore:

- User acquisition
- Product engagement
- User behavior
- Conversion funnel performance
- Add-to-cart activity
- Transactions
- Retention
- Growth trends
- User segmentation
- Product opportunities

The application is designed as an interactive **decision-support platform**, rather than a static dashboard.

---

## 🎯 Business Problem

A product or growth team needs answers to important questions such as:

- How many users are visiting the product?
- How actively are users engaging with the platform?
- Where are users dropping off in the conversion funnel?
- How many users add products to their cart?
- How many users complete transactions?
- Which user groups are most valuable?
- How is user activity changing over time?
- Where are the biggest opportunities for product growth?

Analyzing millions of raw event records manually is inefficient.

This project solves that problem by converting processed event-level data into an interactive application for exploring product and growth performance.

---

## ✨ Key Features

### 📊 Growth Overview

Monitor key product and growth metrics, including:

- Total Visitors
- Total Events
- Add-to-Cart Events
- Transactions

Users can compare current activity with the previous available period.

---

### 👥 User Group Analysis

Analyze different user populations and investigate behavioral differences between user groups.

The application allows users to explore the data dynamically instead of relying on fixed insights.

---

### 📈 Interactive Time Analysis

Users can select different analysis periods to investigate changes in product activity and growth trends over time.

---

### 🛒 Funnel Analysis

Analyze the user journey through important product actions.

The platform helps identify potential drop-off points between stages such as:

1. User visit
2. Product interaction
3. Add to cart
4. Transaction

---

### 📦 Product Engagement Analysis

Investigate how users interact with products and identify patterns in engagement.

---

### 🔍 Interactive Decision Support

The platform is designed for exploration.

Users can adjust filters and controls to investigate different questions without changing the underlying code.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Streamlit | Interactive web application |
| Plotly | Interactive visualizations |
| Git | Version control |
| GitHub | Code hosting |
| Git LFS | Large dataset versioning |
| Streamlit Community Cloud | Application deployment |

---

## 📂 Project Structure

```text
03_Product_Growth_Intelligence/
│
├── .vscode/
│   └── settings.json
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   ├── events_clean.csv
│   │   ├── events_enriched.csv
│   │   ├── session_summary.csv
│   │   └── visitor_summary.csv
│   │
│   └── ...
│
├── notebooks/
│
├── outputs/
│
├── reports/
│
├── sql/
│
├── src/
│
├── tests/
│
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt