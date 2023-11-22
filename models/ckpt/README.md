Place checkpoint files here

model state should be stored like this:

```python
torch.save({'model_state_dict': model.state_dict()}, path/to/pth)
```

see `networks.py` to see the structure of models