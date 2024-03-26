import numpy as np
def get_best_valid_action(qs, move_pieces):
  if len(move_pieces)>1:
    data=[(qs[0,1],0),(qs[0,1],1),(qs[0,2],2),(qs[0,3],3)]
    data.sort(key=lambda x: x[0])
    data.reverse()
    print(data)
    for entry in data:
      if entry[1] in move_pieces:
        return entry[1]
      else:
        continue
  elif len(move_pieces)==1:
    return move_pieces[0]
  else:
    return -1

# Test case with the provided example
qs = np.array([[0.1336262 , -0.05052007, 0.00411009, -0.2452001 ]])
move_pieces = [1, 2]
argmax = get_best_valid_action(qs.copy(), move_pieces.copy())
print(argmax)  # Output: 1 (the first valid index in move_pieces)
