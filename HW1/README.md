We have two versions of the linear regression model:
- **`lr_model1.pkl`**: The original model trained with `x4`.
- **`lr_model2.pkl`**: The new model trained with `x2` and `x4`.
The **current model** is always referenced by **`lr_current.pkl`**. This allows us to easily switch between versions while maintaining access to previous models.
