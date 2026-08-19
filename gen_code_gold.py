#!/usr/bin/env python3
"""Generate a SIZEABLE 'code_gold' corpus of CORRECT canonical solutions to
classic algorithm problems — exactly the pattern greedy eval tests. Emits many
valid variants per problem so (a) the corpus is big enough for the 2048-seq
tiered loader to treat it as a real drillable domain, and (b) greedy learns to
GENERATE correct code robustly (proper base cases, working logic).

Usage: venv/bin/python gen_code_gold.py   →  data/_code_gold/code_gold.jsonl
"""
import json, re
from pathlib import Path

OUT = Path("/home/kenpeter/work/data/_code_gold/code_gold.jsonl")

# One entry per (problem, correct-implementation). Variant expansion below
# renames a small set of benign identifiers to multiply valid solutions.
SOLS = [
# ── strings ──
"def reverse_string(s):\n    return s[::-1]",
"def reverse_string(s):\n    return ''.join(reversed(s))",
"def reverse_string(s):\n    res = ''\n    for ch in s:\n        res = ch + res\n    return res",
"def is_palindrome(s):\n    return s == s[::-1]",
"def is_palindrome(s):\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    return True",
"def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')",
"def word_count(s):\n    return len(s.split())",
"def to_uppercase(s):\n    return s.upper()",
"def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
"def most_frequent_char(s):\n    from collections import Counter\n    return Counter(s).most_common(1)[0][0]",
"def valid_parentheses(s):\n    stack = []\n    pairs = {')': '(', ']': '[', '}': '{'}\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif not stack or stack.pop() != pairs[c]:\n            return False\n    return not stack",
"def is_anagram(s, t):\n    from collections import Counter\n    return Counter(s) == Counter(t)",
"def first_unique_char(s):\n    from collections import Counter\n    counts = Counter(s)\n    for i, c in enumerate(s):\n        if counts[c] == 1:\n            return i\n    return -1",
"def compress_string(s):\n    if not s:\n        return ''\n    res = []\n    cnt = 1\n    for i in range(1, len(s) + 1):\n        if i < len(s) and s[i] == s[i - 1]:\n            cnt += 1\n        else:\n            res.append(s[i - 1] + str(cnt))\n            cnt = 1\n    return ''.join(res)",
# ── arrays ──
"def two_sum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        if target - v in seen:\n            return [seen[target - v], i]\n        seen[v] = i\n    return []",
"class Solution:\n    def twoSum(self, nums, target):\n        hm = {}\n        for i, v in enumerate(nums):\n            need = target - v\n            if need in hm:\n                return [hm[need], i]\n            hm[v] = i\n        return []",
"def max_subarray(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
"def max_product(nums):\n    best = nums[0]\n    cur_max = cur_min = nums[0]\n    for x in nums[1:]:\n        if x < 0:\n            cur_max, cur_min = cur_min, cur_max\n        cur_max = max(x, cur_max * x)\n        cur_min = min(x, cur_min * x)\n        best = max(best, cur_max)\n    return best",
"def contains_duplicate(nums):\n    return len(nums) != len(set(nums))",
"def has_duplicate(nums):\n    seen = set()\n    for x in nums:\n        if x in seen:\n            return True\n        seen.add(x)\n    return False",
"def move_zeros(nums):\n    pos = 0\n    for i in range(len(nums)):\n        if nums[i] != 0:\n            nums[pos], nums[i] = nums[i], nums[pos]\n            pos += 1\n    return nums",
"def remove_duplicates(nums):\n    i = 0\n    for j in range(1, len(nums)):\n        if nums[j] != nums[i]:\n            i += 1\n            nums[i] = nums[j]\n    return i + 1",
"def rotate_array(nums, k):\n    k = k % len(nums)\n    nums[:] = nums[-k:] + nums[:-k]",
"def find_missing_number(nums):\n    n = len(nums)\n    return n * (n + 1) // 2 - sum(nums)",
"def third_max(nums):\n    vals = sorted(set(nums))\n    return vals[-3] if len(vals) >= 3 else vals[-1]",
"def product_except_self(nums):\n    n = len(nums)\n    out = [1] * n\n    left = 1\n    for i in range(n):\n        out[i] = left\n        left *= nums[i]\n    right = 1\n    for i in range(n - 1, -1, -1):\n        out[i] *= right\n        right *= nums[i]\n    return out",
"def intersect(nums1, nums2):\n    from collections import Counter\n    c = Counter(nums1)\n    res = []\n    for x in nums2:\n        if c[x] > 0:\n            res.append(x)\n            c[x] -= 1\n    return res",
"def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    merged = []\n    for lo, hi in intervals:\n        if not merged or lo > merged[-1][1]:\n            merged.append([lo, hi])\n        else:\n            merged[-1][1] = max(merged[-1][1], hi)\n    return merged",
# ── math / number theory ──
"def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
"def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)",
"def tribonacci(n):\n    if n == 0:\n        return 0\n    if n <= 2:\n        return 1\n    a, b, c = 0, 1, 1\n    for _ in range(3, n + 1):\n        a, b, c = b, c, a + b + c\n    return c",
"def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
"def factorial_iter(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
"def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
"def lcm(a, b):\n    return a * b // gcd(a, b)",
"def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out",
"def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0",
"def reverse_integer(x):\n    sign = -1 if x < 0 else 1\n    return sign * int(str(abs(x))[::-1])",
"def sum_digits(n):\n    return sum(int(d) for d in str(n))",
"def count_primes(n):\n    if n < 2:\n        return 0\n    sieve = [True] * n\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(n ** 0.5) + 1):\n        if sieve[i]:\n            for j in range(i * i, n, i):\n                sieve[j] = False\n    return sum(sieve)",
# ── search / sort ──
"def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
"def binary_search_upper(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo",
"def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + mid + quick_sort(right)",
"def merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out",
"def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key\n    return arr",
# ── linked lists ──
"class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    prev = None\n    cur = head\n    while cur:\n        nxt = cur.next\n        cur.next = prev\n        prev = cur\n        cur = nxt\n    return prev",
"def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False",
"def middle_node(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n    return slow",
"def merge_two_lists(l1, l2):\n    dummy = ListNode()\n    tail = dummy\n    while l1 and l2:\n        if l1.val <= l2.val:\n            tail.next = l1; l1 = l1.next\n        else:\n            tail.next = l2; l2 = l2.next\n        tail = tail.next\n    tail.next = l1 or l2\n    return dummy.next",
# ── trees ──
"def max_depth(root):\n    if root is None:\n        return 0\n    return 1 + max(max_depth(root.left), max_depth(root.right))",
"def invert_tree(root):\n    if root is None:\n        return None\n    root.left, root.right = invert_tree(root.right), invert_tree(root.left)\n    return root",
"def is_same_tree(p, q):\n    if p is None and q is None:\n        return True\n    if p is None or q is None:\n        return False\n    return p.val == q.val and is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)",
"def inorder(root):\n    if root is None:\n        return []\n    return inorder(root.left) + [root.val] + inorder(root.right)",
"def preorder(root):\n    if root is None:\n        return []\n    return [root.val] + preorder(root.left) + preorder(root.right)",
"def is_symmetric(root):\n    def is_mirror(a, b):\n        if a is None and b is None:\n            return True\n        if a is None or b is None:\n            return False\n        return (a.val == b.val and is_mirror(a.left, b.right) and is_mirror(a.right, b.left))\n    return is_mirror(root, root)",
# ── dynamic programming ──
"def climb_stairs(n):\n    if n <= 2:\n        return n\n    a, b = 1, 2\n    for _ in range(3, n + 1):\n        a, b = b, a + b\n    return b",
"def coin_change(coins, amount):\n    dp = [amount + 1] * (amount + 1)\n    dp[0] = 0\n    for i in range(1, amount + 1):\n        for c in coins:\n            if c <= i:\n                dp[i] = min(dp[i], dp[i - c] + 1)\n    return -1 if dp[amount] == amount + 1 else dp[amount]",
"def unique_paths(m, n):\n    dp = [1] * n\n    for _ in range(1, m):\n        for j in range(1, n):\n            dp[j] += dp[j - 1]\n    return dp[-1]",
"def length_of_lis(nums):\n    import bisect\n    tails = []\n    for x in nums:\n        i = bisect.bisect_left(tails, x)\n        if i == len(tails):\n            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)",
"def house_robber(nums):\n    if not nums:\n        return 0\n    if len(nums) == 1:\n        return nums[0]\n    prev2, prev1 = 0, nums[0]\n    for x in nums[1:]:\n        prev2, prev1 = prev1, max(prev1, prev2 + x)\n    return prev1",
]

# Safe identifier renames (whole-word; these keep code correct).
RENAMES = [
    ("nums", "arr"), ("nums", "values"), ("nums", "numbers"),
    ("nums", "arrs"), ("s", "text"), ("s", "st"),
    ("target", "goal"), ("target", "tgt"),
    ("arr", "a"), ("arr", "items"),
    ("lo", "low"), ("hi", "high"),
    ("seen", "seen_map"),
]

def _rename(code, old, new):
    if old == new:
        return code
    return re.sub(rf"\b{re.escape(old)}\b", new, code)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    pid = 0
    for si, sol in enumerate(SOLS):
        docs.append({"text": sol, "id": pid, "variant": "base"})
        pid += 1
        for old, new in RENAMES:
            renamed = _rename(sol, old, new)
            if renamed != sol:
                docs.append({"text": renamed, "id": pid, "variant": "r"})
                pid += 1
    with open(OUT, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(docs)} docs to {OUT}")

if __name__ == "__main__":
    main()
