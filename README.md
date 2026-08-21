# 📈 Product & Growth Intelligence Platform

An interactive data analytics and decision-support application built with Python and Streamlit to investigate user behavior, engagement, conversion performance, retention signals, and growth opportunities.

The platform transforms raw user event data into actionable product and growth insights through interactive filtering, user segmentation, funnel analysis, and behavioral metrics.

---

## 🎯 Project Overview

Understanding how users interact with a digital product is essential for improving growth, engagement, and conversion.

This project analyzes user activity and provides an interactive platform where users can investigate:

- User traffic and activity
- New and returning users
- User engagement segments
- Product interaction events
- Add-to-cart behavior
- Transactions
- Conversion funnel performance
- User behavior over selected time periods
- Growth opportunities across different user groups

The application is designed as an interactive decision-support tool rather than a static dashboard.

---

## 💼 Business Problem

Product and growth teams need answers to questions such as:

- How many users are visiting the platform?
- How many of those users are new?
- Which users are highly engaged?
- Which users show low engagement?
- How many users add products to their cart?
- How many users complete transactions?
- Where are users dropping off in the conversion funnel?
- How does user behavior change across different time periods?
- Which user groups provide the strongest growth opportunities?

This platform provides a centralized interface for exploring these questions.

---

## ✨ Key Features

### 1. Interactive Date Analysis

Users can select a custom date range to analyze product performance during a specific period.

### 2. User Segmentation

The application supports analysis across multiple user groups:

- All Users
- Low Engagement Users
- High Engagement Users
- Returning Users
- Converted Users

This allows user behavior to be compared across meaningful segments.

### 3. Growth Overview

The platform provides key performance indicators including:

- Total Visitors
- Total Events
- Add-to-Cart Events
- Transactions

### 4. Conversion Funnel Analysis

The application measures important conversion stages:

**View → Cart → Transaction**

Key conversion metrics include:

- View-to-Cart Conversion
- Cart-to-Transaction Conversion
- Overall Conversion

### 5. New User Analysis

The platform can identify users appearing for the first time during the selected analysis period.

This helps distinguish between:

- New users
- Existing users
- Returning users

### 6. Interactive Decision Support

Instead of presenting only static charts, the application allows the user to dynamically change:

- Analysis period
- User group

All relevant metrics update based on the selected filters.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Plotly | Interactive visualizations |

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
│   └── Dataset files
│
├── notebooks/
│   └── Exploratory analysis notebooks
│
├── outputs/
│   └── Generated outputs
│
├── reports/
│   └── Analysis reports and visualizations
│
├── sql/
│   └── SQL queries and analysis
│
├── src/
│   └── Supporting source code
│
├── tests/
│   └── Testing files
│
├── .gitignore
├── README.md
└── requirements.txt