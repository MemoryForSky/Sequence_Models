class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        win_set = set()
        for i in range(len(nums)):
            if i <= k:
                if nums[i] in win_set:
                    return True
                else:
                    win_set.add(nums[i])
            else:
                win_set.remove(nums[i - k - 1])
                if nums[i] in win_set:
                    return True
                else:
                    win_set.add(nums[i])

        return False


k = 3
nums = [1, 2, 3, 1]
S = Solution()
res = S.containsNearbyDuplicate(nums, k)
print(res)
