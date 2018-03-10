import java.util.*;

public class Main {

    public static void main(String[] args) {
//        ListNode head = new ListNode(1);
//        head.next = new ListNode(1);
//        head.next.next = new ListNode(2);
//        head.next.next.next = new ListNode(2);
//        deleteDuplicates(head);

        int[][] arr = {{1,4,7},
                {2,5,8},
                {3,6,9}};
    }

    //Definition for singly-linked list.
    static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }
    
    //Definition for a binary tree node.
    static class TreeNode {
         int val;
         TreeNode left;
         TreeNode right;
         TreeNode(int x) { val = x; }
    }


    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        return new ArrayList<>();
    }


    public List<Integer> inorderTraversal(TreeNode root) {
        return new ArrayList<>();
    }


    public List<Integer> preorderTraversal(TreeNode root) {
        return new ArrayList<>();
    }


    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        return  new TreeNode(4);
    }


    public TreeNode trimBST(TreeNode root, int L, int R) {
        return  new TreeNode(4);
    }


    public TreeNode sortedArrayToBST(int[] nums) {
         return  new TreeNode(4);
    }


    public String tree2str(TreeNode t) {
        return new String("gh");
    }


    public String decodeString(String s) {
        return "8yu";
    }


    public double myPow(double x, int n) {
        return 0;
    }


    public int findKthLargest(int[] nums, int k) {
        return k;
    }


    public int searcha(int[] nums, int target) {
        return 0;
    }


    public boolean search(int[] nums, int target) {
        return false;
    }


    public boolean judgeCircle(String moves) {
        return true;
    }


    public boolean searchMatrix2(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int i=0,j=matrix[0].length-1;
        while (i < matrix.length) {
            while (j >= 0) {
                if (target == matrix[i][j]) {
                    return true;
                } else if (target > matrix[i][j]) {
                    break;
                }
                j--;
            }
            i++;
        }

        return false;
    }


    public boolean searchMatrix(int[][] matrix, int target) {
        int i = -1;
        if (matrix.length == 0 || matrix[0].length == 0 || target < matrix[0][0] || target > matrix[matrix.length-1][matrix[0].length-1]) {
            return false;
        }
        int rowlen = matrix[0].length;
        while (++i < matrix.length && target > matrix[i][rowlen-1]);
        int left = 0;
        int right = rowlen-1;
        int mid;
        while (left <= right) {
            mid = left + (right - left)/2;
            if (matrix[i][mid] == target) {
                return true;
            } else if (target < matrix[i][mid]) {
                right = mid-1;
            } else {
                left = mid+1;
            }
        }
        return false;
    }


    public int kthSmallest(int[][] matrix, int k) {
        return k;
    }


    public int[] intersect(int[] nums1, int[] nums2) {
        return nums1;
    }


    public int findDuplicate(int[] nums) {
        return 0;
    }


    public ListNode rotateRight(ListNode head, int k) {
        return head;
    }


    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev, next;
        prev = head;
        next = head.next;
        while (next != null) {
            if (prev.val != next.val) {
                prev.next = next;
                prev = prev.next;
            }
            next = next.next;
        }
        prev.next = null;
        return head;
    }


    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        }

        ListNode head, node;
        if (l1.val <= l2.val) {
            head = l1;
            l1 = l1.next;
        } else {
            head = l2;
            l2 = l2.next;
        }
        node = head;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                node.next = l1;
                l1 = l1.next;
            } else {
                node.next = l2;
                l2 = l2.next;
            }
            node = node.next;
        }

        if (l1 != null) {
            node.next = l1;
        } else if (l2 != null) {
            node.next = l2;
        }

        return head;
    }


    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode p1,p2;
        p1 = p2 = head;
        while (p2 != null && p2.next != null) {
            p1 = p1.next;
            p2 = p2.next.next;
            if (p1 == p2) {
                return true;
            }
        }
        return false;
    }


    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode p1,p2;
        p1 = p2 = head;
        while (p2 != null && p2.next != null) {
            p1 = p1.next;
            p2 = p2.next.next;
        }

        ListNode p3 = (p2 == null) ?  p1 : p1.next;
        ListNode prev = null;
        while (p3.next != null) {
            ListNode t = p3.next;
            p3.next = prev;
            prev = p3;
            p3 = t;
        }
        p3.next = prev;

        while (p3 != null) {
            if (p3.val != head.val) {
                return false;
            }
            p3 = p3.next;
            head = head.next;
        }
        return true;
    }


    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p1 = head;
        ListNode p2 = p1.next;
        head.next = null;
        while (p1 != null && p2 != null) {
            ListNode t = p2.next;
            p2.next = p1;
            p1 = p2;
            p2 = t;
        }
        return p1;
    }


    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
       if (headA == null || headB == null) {
           return null;
       }
       int l1 = 0;
       int l2 = 0;
       ListNode temp1 = headA;
       ListNode temp2 = headB;
       while (temp1 != null) {
           l1++;
           temp1 = temp1.next;
       }
       while (temp2 != null) {
           l2++;
           temp2 = temp2.next;
       }
       int diff = Math.abs(l1-l2);
       if (l1 >= l2) {
           temp1 = headA;
           temp2 = headB;
       } else {
           temp1 = headB;
           temp2 = headA;
       }
       while (diff != 0) {
           temp1 = temp1.next;
           diff--;
       }
       while (temp1 != temp2) {
           temp1 = temp1.next;
           temp2 = temp2.next;
       }
       return temp1;

    }


    public String largestNumber(int[] nums) {
        return "dfdf";
    }


    public int maxSubArray(int[] nums) {
        return 3;
    }


    public void sortColors(int[] nums) {

    }


    public boolean judgeSquareSum(int c) {
        return false;
    }


    public boolean checkPossibility(int[] nums) {
        return false;
    }


    public int[] findErrorNums(int[] nums) {
        return nums;
    }


    public int numberOfArithmeticSlices(int[] A) {
        return 0;
    }


    public boolean wordPattern(String pattern, String str) {
        return false;
    }


    public List<String> letterCombinations(String digits) {
        return new ArrayList<>();
    }


    public boolean checkValidString(String s) {
        return true;
    }


    public int calculate(String s) {
        return 0;
    }


    public int findShortestSubArray(int[] nums) {
        return 0;
    }


    public char nextGreatestLetter(char[] letters, char target) {
        return 'c';
    }


    public int arrayPairSum(int[] nums) {
        return 0;
    }


    public String reverseStr(String s, int k) {
        return s;
    }


    public boolean validPalindrome(String s) {
        return false;
    }


    public String complexNumberMultiply(String a, String b) {
        int m,n,x,y,posa,posb;

        if (a.indexOf("+-") > 0) {
            posa = a.indexOf("+-");
            m = Integer.parseInt(a.substring(0, posa));
            n = Integer.parseInt(a.substring(posa+1, a.length() - 1));
        }  else {
            posa = a.indexOf('+');
            m = Integer.parseInt(a.substring(0, posa));
            n = Integer.parseInt(a.substring(posa, a.length() - 1));
        }

        if (b.indexOf("+-") > 0) {
            posb = b.indexOf("+-");
            x = Integer.parseInt(b.substring(0, posb));
            y = Integer.parseInt(b.substring(posb+1, b.length()-1));
        } else {
            posb = b.indexOf('+');
            x = Integer.parseInt(b.substring(0, posb));
            y = Integer.parseInt(b.substring(posb, b.length()-1));

        }

        String ch = (m*y+n*x >= 0) ? "+" : "+-";
        return (m*x-n*y)+ch+Math.abs(m*y+n*x)+"i";
    }


    public String reverseWords(String s) {
        char[] arr = s.toCharArray();
        for(int i=0,j=0; j<arr.length; j++) {
            if (arr[j]==' ' || j==arr.length-1) {
                if (j==arr.length-1)j++;
                for (int m=i,n=j-1; m<n; m++,n--) {
                    char temp = arr[m];
                    arr[m] = arr[n];
                    arr[n] = temp;
                }
                i = j+1;
            }
        }
        return new String(arr);
    }


    public String licenseKeyFormatting(String S, int K) {
        int len = S.length();
        char[] str = new char[len + len/K];
        int count = 0;
        int i,j;
        for (i=len-1, j=str.length-1; i>=0; --i) {
            if (S.charAt(i) != '-') {
                if (count == K) {
                    str[j--] = '-';
                    count = 0;
                }
                str[j--] = S.charAt(i);
                count++;
            }
        }
        return new String(str, j+1, str.length-j-1).toUpperCase();
    }


    public boolean containsDuplicate(int[] nums) {
        Set<Integer> numSet = new HashSet<>();

        for (int i=0; i<nums.length; i++) {
            if (!numSet.add(nums[i])) {
                return true;
            }
        }
        return false;
    }


    public void setZeroes(int[][] matrix) {
        boolean isFirstRowZeros = false;
        boolean isFirstColumnZeros = false;

        for(int j=0; j<matrix[0].length; j++) {
            if(matrix[0][j] == 0) {
                isFirstRowZeros = true;
                break;
            }
        }

        for(int i=0; i<matrix.length; i++) {
            if (matrix[i][0] == 0) {
                isFirstColumnZeros = true;
                break;
            }
        }

        for (int i=1; i<matrix.length; i++)  {
            for(int j=1; j<matrix[i].length; j++) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }

        for (int i=1; i<matrix.length; i++) {
            for (int j=1; j<matrix[0].length; j++) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }


        if (isFirstRowZeros) {
            for (int j=0; j<matrix[0].length; j++) {
                matrix[0][j] = 0;
            }
        }

        if (isFirstColumnZeros) {
            for (int i=0; i<matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
    }


    public int reverse(int x) {
        int num = 0;
        int maxbyten = Integer.MAX_VALUE/10;
        int minbyten = Integer.MIN_VALUE/10;
        int sign = (x<0) ? -1:1;
        int curDigit = 0;
        while(x != 0) {
            curDigit = x%10;
            if ((sign > 0 && maxbyten+curDigit < num+curDigit ) || (sign < 0 && minbyten+curDigit > num+curDigit)) {
                return 0;
            }
            num = num*10 + curDigit;
            x/=10;
        }
        return num;
    }


    public int firstUniqChar(String s) {
        int[] letters = new int[26];

        for(char ch : s.toCharArray()) {
            letters[ch-'a']++;
        }
        for (int i=0; i<s.length(); i++) {
            if (letters[s.charAt(i)-'a'] == 1){
                return i;
            }
        }
        return -1;
    }


    public String countAndSay(int n) {
        String val = "1";
        for (int i=2; i<=n; i++) {
            StringBuilder stringBuilder = new StringBuilder();
            char[] valArray = val.toCharArray();
            int count = 0;
            char prev = valArray[0];
            int j;
            for (j=0; j<valArray.length; j++) {
                if (valArray[j] == prev) {
                    count++;
                } else {
                    stringBuilder.append(count).append(valArray[j - 1]);
                    count = 1;
                    prev = valArray[j];
                }
            }
            val = stringBuilder.append(count).append(valArray[j-1]).toString();
        }
        return val;
    }


    public int majorityElement(int[] nums) {
        int result = nums[0];
        int count = 1;
        for (int i=1; i<nums.length; i++) {
            if (count==0) {
                result = nums[i];
                count++;
            } else if (result == nums[i]) {
                count++;
            } else {
                count--;
            }
        }
        return result;
    }


    public boolean isPowerOfTwo(int n) {
        return n > 0 && (n&(n-1)) == 0;
    }


    public int trailingZeroes(int n) {
        int count = 0;
        for (long i = 5; n / i >= 1; i *= 5) {
            count += n / i;
        }

        return count;
    }


    public int searchInsert(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        int mid = 0;
        while (start <= end) {
            mid = start + (end-start)/2;
            if (target == nums[mid]) {
                return mid;
            }else if(target > nums[mid]) {
                start = mid+1;
            } else {
                end = mid-1;
            }
        }
        if (target > nums[mid]){
            return start;
        } else {
            return mid;
        }
    }


    public int divide(int dividend, int divisor) {
        int sign = 1;
        if(divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1)) {
            return Integer.MAX_VALUE;
        } else if (dividend == Integer.MIN_VALUE && divisor == 1) {
            return Integer.MIN_VALUE;
        } else if((dividend < 0 && divisor >= 0) || (dividend >=0 && divisor < 0)) {
            sign = -1;
        }

        dividend = Math.abs(dividend);
        divisor = Math.abs(divisor);
        if (dividend == Integer.MAX_VALUE && divisor == 1) {
            return Integer.MAX_VALUE;
        }

        int result = 0;
        while(dividend >= divisor) {
            int nShift = 0;
            while(dividend >= (divisor<<nShift)) {
                nShift++;
            }
            result += 1<<(nShift - 1);
            dividend -= divisor<<(nShift - 1);
        }

        return result = (sign < 0) ? -result : result;
    }


    public int pivotIndex(int[] nums) {
        int sum = 0;
        for (int i=0; i<nums.length; i++) {
            sum += nums[i];
        }
        int leftsum = 0;
        for (int i=0; i<nums.length; i++) {
            sum -= nums[i];
            if (leftsum == sum) {
                return i;
            }
            leftsum += nums[i];
        }
        return -1;
    }


    public String[] findWords(String[] words) {
        String[] kb = {"qwertyuiop", "asdfghjkl", "zxcvbnm"};
        int[] arr = new int[26];
        for (int i=0; i<kb.length; i++) {
            for (char ch:kb[i].toCharArray()) {
                arr[ch-'a'] = i;
            }
        }

        List<String> result = new ArrayList<>();
        for (int i=0; i<words.length; i++) {
            String word = words[i].toLowerCase();
            int val = arr[word.charAt(0)-'a'];
            for (char ch:word.toCharArray()) {
                if (val != arr[ch-'a']){
                    val = -1;
                    break;
                }
            }
            if (val != -1) {
                result.add(words[i]);
            }
        }

        return result.toArray(new String[0]);
    }


    public int singleNumbera(int[] nums) {

        int ones = 0;
        int twos = 0;
        int commonBits = 0;
        for (int n : nums) {
            twos = twos|(ones & n);
            ones ^= n;

            commonBits = ones & twos;

            ones &= ~commonBits;
            twos &= ~commonBits;
        }
        return ones;
    }


    public int[] singleNumber(int[] nums) {
        int xored = 0;
        for (int n:nums) {
            xored ^= n;
        }

        int rightMostBit = xored & ~(xored-1);
        int[] result = new int[2];
        for (int n:nums) {
            if ((n&rightMostBit) > 0) {
                result[0] ^= n;
            } else {
                result[1] ^= n;
            }
        }
        return result;
    }


    public List<Integer> getRow(int rowIndex) {
        List<Integer> numList = new ArrayList<>();

        numList.add(1);
        for (int i=1; i<=rowIndex; i++) {
            for (int j=numList.size()-1; j>0; j--) {
                numList.set(j, numList.get(j) + numList.get(j-1));
            }
            numList.add(1);
        }
        return numList;
    }


    public ListNode removeElements(ListNode head, int val) {
        while (head != null && head.val == val) {
            head = head.next;
        }
        ListNode temp = head;
        while (temp != null && temp.next != null) {
            if (temp.next.val == val) {
                temp.next = temp.next.next;
            } else {
                temp = temp.next;
            }
        }
        return  head;
    }


    public boolean detectCapitalUse(String word) {
        if (word.length()==1 || word.equals(word.toUpperCase()) || word.substring(1).equals(word.substring(1).toLowerCase())) {
            return true;
        }
        return false;
    }


    public int romanToInt(String s) {
        Map<Character, Integer> valMap = new HashMap<>();
        valMap.put('M', 1000);
        valMap.put('D', 500);
        valMap.put('C', 100);
        valMap.put('L', 50);
        valMap.put('X', 10);
        valMap.put('V', 5);
        valMap.put('I', 1);

        int result = 0;
        int diff = 0;
        for(int i=0; i<s.length();) {
            if (i != s.length()-1 &&(diff = valMap.get(s.charAt(i+1)) - valMap.get(s.charAt(i))) > 0) {
                result += diff;
                i += 2;
            } else {
                result += valMap.get(s.charAt(i));
                i++;
            }
        }
        return result;
    }


    public int countPrimeSetBits(int L, int R) {
        boolean[] isPrime = new boolean[33];
        Arrays.fill(isPrime, true);
        isPrime[0] = isPrime[1] = false;
        for (int i=2; i*i <= 32; i++) {
            if (isPrime[i]) {
                for (int j=2*i; j<=32; j+=i) {
                    isPrime[j] = false;
                }
            }
        }

        int result = 0;
        for (int i=L; i<=R; i++) {
            if(isPrime[Integer.bitCount(i)]) {
                result++;
            }
        }
        return result;
    }


    public boolean isToeplitzMatrix(int[][] matrix) {
        for (int r=1; r<matrix.length; r++) {
            for (int c=1; c<matrix[r].length; c++) {
                if (matrix[r][c] != matrix[r-1][c-1]) {
                    return false;
                }
            }
        }
        return true;
    }


    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> results = new ArrayList<>();
        boolean flag;
        for (int i=left; i<=right; i++) {
            flag = true;
            int n = i;
            while (n != 0){
                int digit = n%10;
                if (digit == 0 || i%digit != 0) {
                    flag = false;
                    break;
                }
                n /= 10;
            }
            if (flag) {
                results.add(i);
            }
        }
        return results;
    }


    public int[][] matrixReshape(int[][] nums, int r, int c) {
        if (nums.length * nums[0].length != r*c) {
            return nums;
        }
        int[][] reshape = new int[r][c];
        int pos = 0;
        int m=0,n=0;
        for (int i=0; i<nums.length; i++) {
            for (int j=0; j<nums[i].length; j++) {
                reshape[m][n] = nums[i][j];
                pos++;
                m = pos/c;
                n = pos%c;
            }
        }
        return reshape;
    }


    public boolean isHappy(int n) {
        int num = n;
        Set<Integer> numbers = new HashSet<Integer>(Arrays.asList(0,2,4,16,20,37,42,58,89,145));
        while(n != 1) {
            int sum = 0;
            while(n != 0) {
                sum += (n%10)*(n%10);
                n /= 10;
            }
            if(numbers.contains(sum)) return false;
            n=sum;
        }
        return true;
    }


    public boolean checkPerfectNumber(int num) {
        int sum = 1;
        for (int i=2; i*i<=num; i++) {
            if (num%i == 0) {
                sum += i+num/i;
            }
        }

        return (sum == num);
    }


    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> numSet = new HashSet<>();
        for (int i=0; i<nums.length; i++) {
            if (!numSet.add(nums[i])) {
                return true;
            }

            if (i>=k) {
                numSet.remove(nums[i - k]);
            }
        }
        return false;
    }


    public String reverseVowels(String s) {
        HashSet<Character> vowels = new HashSet<>(Arrays.asList('a','e','i','o','u'));
        int len = s.length()-1;
        char[] str = s.toCharArray();
        boolean isLeftVowel,isRightVowel;
        for (int i=0,j=len; i<j;) {
            isLeftVowel = vowels.contains(str[i]);
            isRightVowel = vowels.contains(str[j]);
            if (isLeftVowel && isRightVowel) {
                char temp = str[i];
                str[i] = str[j];
                str[j] = temp;
            }
            if (!isLeftVowel) {
                i++;
            }
            if (!isRightVowel) {
                j--;
            }
        }
        return new String(str);
    }


    public int missingNumber(int[] nums) {
        boolean[] isPresent = new boolean[nums.length+1];
        for (int i:nums) {
            isPresent[i] = true;
        }
        int missing = 0;
        for (int i=0; i<isPresent.length; i++){
            if (!isPresent[i]) {
                missing = i;
                break;
            }
        }
        return missing;
    }


    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        } else if (s.equals(t)) {
            return true;
        }
        int[] chars = new int[128];
        for (char ch:s.toCharArray()) {
            chars[ch]++;
        }
        for (char ch:t.toCharArray()) {
            chars[ch]--;
        }
        for (int ch:chars){
            if (ch != 0) return false;
        }
        return true;
    }


    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        } else if (s.equals(t)) {
            return true;
        }
        Map<Character, Character> letterMap = new HashMap<>();
        for (int i=0; i<s.length(); i++) {
            if (!letterMap.containsKey(s.charAt(i))) {
                if (letterMap.containsValue(t.charAt(i))) {
                    return false;
                }
                letterMap.put(s.charAt(i), t.charAt(i));
            } else if (letterMap.get(s.charAt(i)) != t.charAt(i)) {
                return false;
            }
        }
        return true;
    }
}
