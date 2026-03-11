**Total Customers** 

    = DISTINCTCOUNT(bank_churn_predictions[CustomerId])

**Target_Customer_Color**

    = IF(
        MAX('bank_churn_predictions'[Balance]) >= 100000 && MAX('bank_churn_predictions'[Churn_Risk_Percentage]) >= [Risk Threshold Value],
        "#E63946", // A bright, urgent Red for the high-risk customers
        "#CCCCCC"  // A muted Grey for everyone else
    )

**Revenue_At_Risk**

    = CALCULATE(
        SUM('bank_churn_predictions'[Balance]),
        FILTER(
            'bank_churn_predictions',
            'bank_churn_predictions'[Churn_Risk_Percentage] >= [Risk Threshold Value]
        )
    )

**Critical_Flight_Risks**

    = CALCULATE(
        COUNT('bank_churn_predictions'[CustomerId]),
        FILTER(
            'bank_churn_predictions',
            'bank_churn_predictions'[Churn_Risk_Percentage] >= [Risk Threshold Value]
        )
    )

**Churn_Rate_%** 

    = AVERAGE('bank_churn_predictions'[Predicted_Churn])
