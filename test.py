import numpy as np
def get_best_valid_action(qs, move_pieces):
  if len(move_pieces) > 1:
    data={qs[0, 0]:0, qs[0, 1]:1, qs[0, 2]:2, qs[0, 3]:3}
    keys=list(data.keys())
    keys.sort()
    keys.reverse()
    sorted_data = {i: data[i] for i in keys}
    print(sorted_data)
    for entry in data:
      if any(entry[1] == piece for piece in move_pieces):  # Use any() to check if any element in move_pieces matches entry[0]
        return entry[1]
  elif len(move_pieces) == 1:
    return move_pieces[0]
  else:
    return -1

# Test case with the provided example
qs = np.array([[0.01336262 , 0.05052007, -1, 0.2452001 ]])
move_pieces = [1, 2]
argmax = get_best_valid_action(qs.copy(), move_pieces.copy())
print(argmax)  # Output: 1 (the first valid index in move_pieces)
