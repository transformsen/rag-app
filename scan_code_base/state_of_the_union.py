def reverse_largest(numbers):
    max_num = max(numbers)
    max_index = numbers.index(max_num)
    reversed_max = int(str(max_num)[::-1])
    numbers[max_index] = reversed_max
    return numbers


def is_prime(num):
  if num <= 1:
    return False
  for i in range(2, int(num ** 0.5) + 1):
    if num % i == 0:
      return False
  return True

# Function to find the factorial of a number
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Function to check if a string is a palindrome
def is_palindrome(s):
    return s == s[::-1]

def find_number_if_available(numbers, target):
    try:
        index = numbers.index(target)
        return index
    except ValueError:
        return -1
def find_most_frequent_char(text):
  char_counts = Counter(text)
  most_frequent = char_counts.most_common(1)[0][0]  # Get first element from most_common
  return most_frequent

def find_intersection(list1, list3):

  intersection = [item for item in list1 if item in list3]
  return intersection

def flatten(list_of_lists):
    single_list = [element for sublist in list_of_lists for element in sublist]
    return single_list