import numpy as np
import pandas as pd
import openai
from typing import Dict, List, Any, Tuple

import numpy as np

def create_sample_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data for training a model, enforcing stronger correlation
    between credit score, income, and coverage so that higher credit scores
    more reliably lead to approvals (and thus more intuitive SHAP values).

    Args:
        n_samples: Number of samples to generate.

    Returns:
        A tuple (X, y) where:
            X is a 2D NumPy array of shape (n_samples, 5):
                [credit_score, age, income, claims_history, coverage_amount].
            y is a 1D NumPy array of binary labels (0 or 1), indicating approval.
    """
    np.random.seed(42)

    credit_scores = np.random.normal(loc=700, scale=100, size=n_samples).clip(300, 850)

    ages = np.random.normal(loc=40, scale=15, size=n_samples).clip(18, 100)

    claims_history = np.random.poisson(lam=1, size=n_samples).clip(0, 10)


    base_incomes = np.random.lognormal(mean=11, sigma=1, size=n_samples)
    

    scale_factor = (credit_scores - 300) / (850 - 300)
    incomes = base_incomes * (1 + 2 * scale_factor)
    incomes = incomes.clip(0, 1_000_000)


    coverage_multiplier = np.random.uniform(0.5, 2.0, n_samples)

    coverage_multiplier *= (1 + 0.5 * scale_factor)
    coverage_amounts = (incomes * coverage_multiplier).clip(0, 1_000_000)


    X = np.column_stack([
        credit_scores,
        ages,
        incomes,
        claims_history,
        coverage_amounts
    ])


    y = (
        (credit_scores > 650) &
        (incomes > 30_000) &
        (claims_history < 3) &
        (coverage_amounts < incomes * 5) &
        (ages < 40)
    ).astype(int)

    return X, y

def preprocess_input(data: Dict[str, float]) -> np.ndarray:
    """
    Preprocess input data for model prediction.
    
    Args:
        data: Dictionary of input features
        
    Returns:
        Numpy array of preprocessed features
    """
    return np.array([[ 
        data['credit_score'],
        data['age'],
        data['income'],
        data['claims_history'],
        data['coverage_amount']
    ]])

def generate_explanation_letter(
    applicant_name: str,
    policy_number: str,
    decision: str,
    shap_values: np.ndarray,
    features: List[str],
    feature_values: Dict[str, Any]
) -> str:
    """
    Generate an explanation letter using OpenAI API.
    
    Args:
        applicant_name: Name of the applicant
        policy_number: Policy number
        decision: Approval decision
        shap_values: SHAP values for feature importance
        features: List of feature names
        feature_values: Dictionary of feature values
        
    Returns:
        Generated explanation letter
    """

    feature_importance = list(zip(features, shap_values[0][0]))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    

    top_factors = []
    for feature, importance in feature_importance[:3]:
        value = feature_values[feature.lower().replace(' ', '_')]
        impact = "positively" if importance > 0 else "negatively"
        top_factors.append(f"{feature} ({value}) impacted the decision {impact}")
    

    prompt = f"""
    Write a professional underwriting decision letter with the following details:
    
    Applicant: {applicant_name}
    Policy Number: {policy_number}
    Decision: {decision}
    
    Top factors influencing the decision:
    {chr(10).join('- ' + factor for factor in top_factors)}
    
    The letter should:
    1. Start with a professional greeting
    2. State the decision clearly
    3. If they were approved, explaing the main positive factors that influenced the decision in a congrat
    4. If they were rejected, explaining the main negative factors that influenced the decision in an understanding way. 
    5. Provide constructive feedback and next steps
    6. End with a professional closing
    
    Keep the tone professional but empathetic.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "You are a professional underwriter writing an explanation letter."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print('done')
        next_steps = "If you would like to discuss this decision or explore options for improving your application, please don't hesitate to contact our office." if decision == "Rejected" else "We look forward to providing you with excellent service."
        
        return f"""
        Dear {applicant_name},

        Re: Policy Number {policy_number}

        We are writing to inform you about the status of your insurance policy application. After careful review, your application has been {decision.lower()}.

        The key factors that influenced this decision include:
        {chr(10).join('- ' + factor for factor in top_factors)}

        {next_steps}

        Sincerely,
        Your Insurance Team
        """ 