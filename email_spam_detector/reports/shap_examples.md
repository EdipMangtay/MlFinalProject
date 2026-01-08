# SHAP Explanation Examples

This report demonstrates explainability for spam email classification.

## Example 1: Spam Email

**Text**:
```
subject: naturally irresistible your corporate identity lt is really hard to recollect a company : the market is full of suqgestions and the information isoverwhelminq ; but a good catchy logo , stylish statlonery and outstanding website will make the task much easier . we do not promise that havinq ordered a iogo your company will automaticaily become a world ieader : it isguite ciear that without good products , effective business organization and practicable aim it will be hotat nowadays mark...
```

**Top Contributing Tokens (towards spam classification):**

- 🔴 `content`: 0.1509
- 🔴 `logo`: 0.1363
- 🔴 `marketing`: 0.0991
- 🔴 `website`: 0.0666
- 🔴 `guaranteed`: 0.0655
- 🔴 `100`: 0.0649
- 🔴 `logos`: 0.0637
- 🔴 `subject`: 0.0584
- 🔴 `effective`: 0.0492
- 🔴 `extra`: 0.0486

## Example 2: Ham Email

**Text**:
```
subject: hello guys , i ' m " bugging you " for your completed questionnaire and for a one - page bio / statement on your thoughts on " business edu and the new economy " . if my records are incorrect please re - ship your responses to me . i want to put everything together next week so that i can ship it back to everyone . the questionnaire is attached as well as copies of the bio pages for michael froehls and myself ( two somewhat different approaches ) . the idea of the latter is just to intr...
```

**Top Contributing Tokens (towards ham classification):**

- 🟢 `doc`: -0.1083
- 🔴 `http`: 0.0993
- 🟢 `copies`: -0.0680
- 🔴 `subject`: 0.0665
- 🔴 `po box`: 0.0618
- 🟢 `edu`: -0.0563
- 🟢 `john`: -0.0562
- 🔴 `introduce`: 0.0498
- 🔴 `po`: 0.0415
- 🔴 `baylor`: 0.0399

## Explanation Method

The explanation uses model coefficients (for Logistic Regression and Linear SVM) 
or log probability differences (for Naive Bayes). 
Positive impact values indicate tokens that push towards spam classification, 
while negative values indicate tokens that push towards ham classification.

## Code Example

```python
from src.explain import explain_text

# Explain a text
result = explain_text('your email text here', model_name='best', top_k=5)

# Print top tokens
for token_info in result['tokens']:
    print(f"{token_info['token']}: {token_info['impact']:.4f}")
```
