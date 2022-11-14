from tensordict import TensorDict
import torch
# t = TensorDict({'a.b': torch.randn(3), 'a': {'b': torch.randn(3)}}, [])

# # Should work
t = TensorDict(
    {
        'a.b.c': torch.randn(3), 
        'a.b.x': torch.randn(3), 
        'a': {
            'b': {
                'c': torch.randn(3)
            }
        },
        'g': {
            'd': {
                'f': torch.randn(3)
            }
        }
    }, 
    [])

# Should break
# t = TensorDict(
#     {
#         'a.b.c': torch.randn(3), 
#         'a': {
#             'b': {
#                 'c': torch.randn(3)
#             }
#         },
#         'g': {
#             'd': {
#                 'f': torch.randn(3)
#             }
#         }
#     }, 
#     [])

# t = TensorDict({'a.b.c': torch.randn(3), 'a': {'b': {'c': torch.randn(3)}}}, [])
# t = TensorDict({'a': {'b': {'c': torch.randn(3)}}}, [])
# t = TensorDict({'a': {'b': {'c': torch.randn(3)}}}, [])



# print(t.unflatten_keys('.'))
print(t.flatten_keys('.'))