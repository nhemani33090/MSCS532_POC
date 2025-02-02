# **E-commerce Recommendation System - Proof of Concept (PoC)**

## **📖 Overview**
This repository contains the **proof-of-concept (PoC) implementation** for an **e-commerce recommendation system** that provides personalized product recommendations based on user interactions and product similarities. The system utilizes **three key data structures**:

1. **UserRegistry** (Hash Table) - Manages user profiles and preferences.  
2. **InteractionStore** (Matrix Representation) - Tracks user-product interactions.  
3. **ProductMatcher** (KD-Tree) - Finds similar products based on feature vectors.  

This PoC serves as a **foundation** for building a scalable recommendation system and will be further optimized in later phases.

---

## **📌 How to Download and Run the Code**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/nhemani33090/MSCS532_POC
cd MSCS532_POC
```

### **2️⃣ Run the PoC Implementation**
Ensure you have **Python3 installed**, then execute:
```bash
python3 poc.py
```

### **3️⃣ Expected Output**
The script runs **various test cases**, including:
- **UserRegistry Tests** (Adding, Retrieving, Deleting Users)
- **InteractionStore Tests** (Storing & Retrieving Ratings)
- **ProductMatcher Tests** (Finding Similar Products)

---