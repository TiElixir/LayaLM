from laya import Router

router = Router()

text = "My payment was deducted twice. I need a refund."

questions = {
    "category": {
        "type": "choice",
        "instructions": "Classify this customer message.",
        "criteria": {
            "billing": "Payment or refund issues",
            "technical": "Technical problems",
            "other": "Other issues",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Is this message urgent?",
    },
}

result = router.predict(text, questions)
print(result["answers"])