import algorithms.Recursion;
import algorithms.Searching;
import algorithms.Sorting;
import data_structure.Heap;
import data_structure.MyLinkedList;
import data_structure.SimpleQueue;
import data_structure.SimpleStack;

import java.util.*;

public class Main {

    public static void main(String[] args) {
//        ListNode head = new ListNode(1);
//        head.next = new ListNode(1);
//        head.next.next = new ListNode(2);
//        head.next.next.next = new ListNode(2);
//        deleteDuplicates(head);

        int[][] arr = {{1,5,9},
                {10,11,13},
                {12,13,15}};

        int[] nums = {73,74,75,71,69,72,76,73};
//        System.out.println(Arrays.toString(dailyTemperatures(nums)));
        frequencySort("tree");

//        showRecursion();

//        showSorting();

//        showSearching();

//        showGraph();

//        showMyLinkedList();

//        showStack();

//        showQueue();

//        showHeap();

    }

    public static void showHeap() {
        Heap heap = new Heap(1);
        int[] array = {3,5,2,4,8,1,0,6,9,7};
        for (int num : array) {
            heap.add(num);
            System.out.println(heap.toString());
        }
        heap.poll();
        System.out.println(heap.toString());
        heap.poll();
        System.out.println(heap.toString());
    }

    public static void showQueue() {
        SimpleQueue queue = new SimpleQueue();
        queue.push(1);
        int num = queue.pop();
        System.out.println(num + " ");
        System.out.println(queue.isEmpty() + " ");
        queue.push(3);
        num = queue.peek();
        System.out.println(num + " ");
        queue.push(4);
        num = queue.pop();
        System.out.println(num + " ");
        num = queue.pop();
        System.out.println(num + " ");
        num = queue.peek();
        System.out.println(num + " ");
    }

    public static void showStack() {
        SimpleStack stack = new SimpleStack();
        stack.push(1);
        int num = stack.pop();
        System.out.println(num + " ");
        System.out.println(stack.isEmpty() + " ");
        stack.push(3);
        num = stack.peek();
        System.out.println(num + " ");
        stack.push(4);
        num = stack.pop();
        System.out.println(num + " ");
        num = stack.pop();
        System.out.println(num + " ");
        num = stack.peek();
        System.out.println(num + " ");
    }

    public static void showGraph() {
        data_structure.Graph graph = new data_structure.Graph("graph.txt");
        graph.traverseDFS();
        graph.traverseBFS();
    }

    public static void showMyLinkedList() {
        MyLinkedList MyLinkedList = new MyLinkedList(new int[]{5,4,3,2,1});
        MyLinkedList.search(1);
        MyLinkedList.prepend(0);
        MyLinkedList.search(-1);
        MyLinkedList.appendNode(6);
        MyLinkedList.search(-1);
        MyLinkedList.deleteNode(4);
        MyLinkedList.search(-1);
    }

    public static void showSearching() {
        int[] arr = {3,1,4,6,7,8,2,9,0,5,9};
        Sorting.quickSort(arr);
        System.out.println("Recursive Search Index : " + Searching.binarySearchR(arr, 9));
        System.out.println("Iterative Search Index : " + Searching.binarySearchI(arr, 7));
    }

    public static void showSorting() {
        int[] arr = {3,1,4,6,7,8,2,9,0,5,0,9};

//        Sorting.bubbleSort(arr);
//        System.out.println(Arrays.toString(arr));
//        Sorting.mergeSort(arr);
//        System.out.println(Arrays.toString(arr));
//        Sorting.quickSort(arr);
//        System.out.println(Arrays.toString(arr));
//        Sorting.heapSort(arr);
//        System.out.println(Arrays.toString(arr));
        String[] words = {"11", "12", "10"};
        Arrays.sort(words, new Comparator<String>() {
            @Override
            public int compare(String a, String b) {
                return (Integer.parseInt(a) - Integer.parseInt(b));
            }
        });
        System.out.println(Arrays.toString(words));
    }

    public static void showRecursion() {
        for(int i=0; i<47; i++) {
            System.out.print(Recursion.fibR(i)+" ");
        }
        System.out.println();
        for(int i=0; i<47; i++) {
            System.out.print(Recursion.fibM(i)+" ");
        }
        System.out.println();
        for(int i=0; i<47; i++) {
            System.out.print(Recursion.fibMDp(i)+" ");
        }
        System.out.println();
        for(int i=0; i<47; i++) {
            System.out.print(Recursion.fibMI(i)+" ");
        }
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


    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode node = root;
        int result = 0;
        while(!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.add(node);
                node = node.left;
            } else {
                node = stack.pop();
                if (--k == 0) {
                    result = node.val;
                }
                node = node.right;
            }
        }
        return result;
    }


    public int[] findMode(TreeNode root) {
        return new int[]{7,9};
    }


    public TreeNode invertTree(TreeNode root) {
        invertChild(root);
        return root;
    }


    public void invertChild(TreeNode node) {
        if (node == null) {
            return;
        }
        TreeNode tempNode = node.left;
        node.left = node.right;
        node.right = tempNode;
        invertChild(node.left);
        invertChild(node.right);
    }


    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> averageList = new ArrayList<>();
        double sum = 0;
        int count = 0;
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();

        if (root != null) {
            current.add(root);
        }
        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            sum += node.val;
            count++;
            if (node.left != null) {
                next.add(node.left);
            }
            if (node.right != null) {
                next.add(node.right);
            }
            if (current.isEmpty()) {
                averageList.add(sum/count);
                sum = 0;
                count = 0;
                current = next;
                next = new LinkedList<>();
            }
        }

        return averageList;
    }


    public List<Integer> largestValues(TreeNode root) {
        List<Integer> maxList = new ArrayList<>();
        int max = Integer.MIN_VALUE;
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();

        if (root != null) {
            current.add(root);
        }
        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            if (max < node.val) {
                max = node.val;
            }
            if (node.left != null) {
                next.add(node.left);
            }
            if (node.right != null) {
                next.add(node.right);
            }
            if (current.isEmpty()) {
                maxList.add(max);
                max = Integer.MIN_VALUE;
                current = next;
                next = new LinkedList<>();
            }
        }

        return maxList;
    }


    public int countSubstrings(String s) {
        return 0;
    }


    public boolean isSymmetric(TreeNode root) {
        return false;
    }


    public boolean hasPathSum(TreeNode root, int sum) {
        return false;
    }


    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        return new ArrayList<>();
    }


    public int maxPathSumHelper(TreeNode root, int[] maxPath) {
        if (root == null) {
            return 0;
        }

        int leftSum = maxPathSumHelper(root.left, maxPath);
        int rightSum = maxPathSumHelper(root.right, maxPath);

        int current = Math.max(root.val, Math.max(root.val + leftSum, root.val + rightSum));
        maxPath[0] = Math.max(maxPath[0] ,Math.max(current, leftSum + root.val + rightSum));
        return current;
    }


    public int maxPathSum(TreeNode root) {
        int[] maxPath = {Integer.MIN_VALUE};
        return maxPath[0];
    }


    public List<Integer> rightSideView(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>();
        List<Integer> rightList = new ArrayList<>();

        if (root != null) {
            queue.add(root);
        }

        while (!queue.isEmpty()) {
            int queueSize = queue.size();
            for (int i=0; i<queueSize; i++) {
                TreeNode node = queue.poll();
                if (i == 0) {
                    rightList.add(node.val);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
                if (node.left != null) {
                    queue.add(node.left);
                }
            }
        }

        return rightList;
    }


    public List<List<Integer>> levelOrderBottom(TreeNode root) {
            List<List<Integer>> orderList = new ArrayList<>();
            LinkedList<TreeNode> current = new LinkedList<>();
            LinkedList<TreeNode> next = new LinkedList<>();
            List<Integer> currentVals = new ArrayList<>();

            if (root != null) {
                current.add(root);
            }
            while (!current.isEmpty()) {
                TreeNode node = current.poll();
                currentVals.add(node.val);
                if (node.left != null) {
                    next.add(node.left);
                }
                if (node.right != null) {
                    next.add(node.right);
                }
                if (current.isEmpty()) {
                    orderList.add(0, currentVals);
                    currentVals = new ArrayList<>();
                    current = next;
                    next = new LinkedList<>();
                }
            }

            return orderList;
    }


    public int findTilt(TreeNode root) {
        return 0;
    }


    public TreeNode convertBST(TreeNode root) {
        return root;
    }


    public int treeHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftHeight = treeHeight(root.left);
        int rightHeight = treeHeight(root.right);
        return 1 + Math.max(leftHeight, rightHeight);
    }


    public int treeHeight(TreeNode root, int[] diameter) {
        if (root == null) {
            return 0;
        }
        int leftHeight = treeHeight(root.left, diameter);
        int rightHeight = treeHeight(root.right, diameter);
        diameter[0] = Math.max(diameter[0], 1+leftHeight+rightHeight);
        return 1 + Math.max(leftHeight, rightHeight);
    }


    public int diameterOfBinaryTree(TreeNode root) {
        return Math.max(0, treeHeight(root, new int[1])-1);
    }


    public int findBottomLeftValue(TreeNode root) {
        return 0;
    }


    public List<String> binaryTreePaths(TreeNode root) {
        return new ArrayList<>();
    }


    public int findSecondMinimumValue(TreeNode root) {
        return 0;
    }


    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == q) {
            return true;
        } else if (p == null || q == null || p.val != q.val) {
            return false;
        }

        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }


    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> orderList = new ArrayList<>();
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();
        List<Integer> currentVals = new ArrayList<>();

        if (root != null) {
            current.add(root);
        }
        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            currentVals.add(node.val);
            if (node.left != null) {
                next.add(node.left);
            }
            if (node.right != null) {
                next.add(node.right);
            }
            if (current.isEmpty()) {
                orderList.add(currentVals);
                currentVals = new ArrayList<>();
                current = next;
                next = new LinkedList<>();
            }
        }

        return orderList;
    }


    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> orderList = new ArrayList<>();
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();
        List<Integer> currentVals = new ArrayList<>();

        if (root != null) {
            current.add(root);
        }
        int levelCount = 0;
        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            if (levelCount%2 == 0) {
                currentVals.add(node.val);
            } else {
                currentVals.add(0, node.val);
            }
            if (node.left != null) {
                next.add(node.left);
            }
            if (node.right != null) {
                next.add(node.right);
            }
            if (current.isEmpty()) {
                orderList.add(currentVals);
                currentVals = new ArrayList<>();
                current = next;
                next = new LinkedList<>();
            }
        }

        return orderList;
    }


    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> inorderList = new ArrayList<>();
        if (root == null) {
            return inorderList;
        }
        TreeNode node = root;
        Stack<TreeNode> nodeStack = new Stack<>();
        while (node != null) {
            nodeStack.push(node);
            node = node.left;
        }
        while (!nodeStack.isEmpty()) {
            node = nodeStack.pop();
            inorderList.add(node.val);
            if (node.right != null) {
                node = node.right;
                while (node != null) {
                    nodeStack.push(node);
                    node = node.left;
                }
            }
        }
        return inorderList;
    }


    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> orderList = new ArrayList<>();
        if (root == null) {
            return orderList;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            orderList.add(node.val);
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return orderList;
    }


    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> orderList = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        if (root != null) {
            stack.push(root);
        }
        while (!stack.isEmpty()) {
            TreeNode node = stack.peek();
            boolean isLeaf = true;
            if (node.right != null) {
                stack.push(node.right);
                isLeaf = false;
                node.right = null;
            }
            if (node.left != null) {
                stack.push(node.left);
                isLeaf = false;
                node.left =null;
            }

            if (isLeaf) {
                orderList.add(node.val);
                stack.pop();
            }
        }
        return orderList;
    }


    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null) {
            return t2;
        } else if (t2 == null) {
            return t1;
        }

        t1.val += t2.val;
        t1.left = mergeTrees(t1.left, t2.left);
        t1.right = mergeTrees(t1.right, t2.right);

        return t1;
    }


    public TreeNode trimBST(TreeNode root, int L, int R) {
        return  new TreeNode(4);
    }


    public TreeNode sortedArrayToBST(int[] nums) {
         return  new TreeNode(4);
    }


    public int maxDepth(TreeNode root) {
        int depth = 0;
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();
        if (root != null) {
            current.add(root);
        }

        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            if (node.left != null) {
                next.add(node.left);
            }
            if (node.right != null) {
                next.add(node.right);
            }
            if (current.isEmpty()) {
                current = next;
                next = new LinkedList<>();
                depth++;
            }
        }
        return depth;
    }


    public int minDepth(TreeNode root) {
        int depth = 0;
        LinkedList<TreeNode> current = new LinkedList<>();
        LinkedList<TreeNode> next = new LinkedList<>();
        if (root != null) {
            current.add(root);
        }

        while (!current.isEmpty()) {
            TreeNode node = current.poll();
            boolean flag = true;
            if (node.left != null) {
                next.add(node.left);
                flag = false;
            }
            if (node.right != null) {
                next.add(node.right);
            } else if (flag) {
                depth++;
                break;
            }
            if (current.isEmpty()) {
                current = next;
                next = new LinkedList<>();
                depth++;
            }
        }
        return depth;
    }


    public String tree2str(TreeNode t) {
        if (t == null) {
            return "";
        }
        String right = (t.right != null) ? "(" + tree2str(t.right) + ")" : "";
        String left = (t.left != null) ? "(" + tree2str(t.left) + ")" : ((t.right != null) ? "()" : "");
        return t.val + left + right;
    }


    public static String frequencySort(String s) {
        int[] count = new int[256];
        for (char c : s.toCharArray()) {
            count[c]++;
            System.out.println(c + " " + count[c]);
        }
        Arrays.sort(count);
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < count.length; i++) {
            if (count[i] > 0) {
                System.out.println(i + " " + i + " " + count[i]);
                char[] arr = new char[count[i]];
                Arrays.fill(arr, (char) i);
                stringBuilder.append(arr);
            }
        }

        return stringBuilder.toString();
    }


    public int[] intersect2(int[] nums1, int[] nums2) {
        return new int[]{3, 4};
    }


    public ListNode removeNthFromEnd(ListNode head, int n) {
        return head;
    }


    public int findPairs(int[] nums, int k) {
        return 0;
    }


    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> wordMap = new HashMap<>();
        for (String word : words) {
            if (wordMap.containsKey(word)) {
                wordMap.put(word, wordMap.get(word) + 1);
            } else {
                wordMap.put(word, 1);
            }
        }

        PriorityQueue<Map.Entry<String, Integer>> wordHeap = new PriorityQueue<>(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> entry1, Map.Entry<String, Integer> entry2) {
                int val = entry1.getValue() - entry2.getValue();
                return (val == 0) ? entry2.getKey().compareTo(entry1.getKey()) : val;
            }
        });

        for (Map.Entry<String, Integer> entry : wordMap.entrySet()) {
            wordHeap.offer(entry);
            if (wordHeap.size() > k) {
                wordHeap.poll();
            }
        }

        List<String> topKFrequent = new ArrayList<>();
        while (wordHeap.size() > 0) {
            topKFrequent.add(wordHeap.poll().getKey());
        }
        Collections.reverse(topKFrequent);
        return topKFrequent;
    }


    public int[] maxSlidingWindow(int[] nums, int k) {
        return new int[]{3, 2};
    }


    public String decodeString(String s) {
        return "8yu";
    }


    public double myPow(double x, int n) {
        return 0;
    }


    public int maxAreaOfIsland(int[][] grid) {
        return 0;
    }


    public List<String> letterCasePermutation(String S) {
        return new ArrayList<>();
    }


    public int totalHammingDistance(int[] nums) {
        int count = 0;
        for (int i=0; i<32; i++) {
            int ones = 0;
            int pos = 1<<i;
            for (int j=0; j<nums.length; j++) {
                if ((nums[j] & pos) > 0) {
                    ones++;
                }
            }
            count += ones * (nums.length - ones);
        }
        return count;
    }


    public int titleToNumber(String s) {
        return 0;
    }


    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> numMap = new HashMap<>();
        for (int num : nums) {
            if (numMap.containsKey(num)) {
                numMap.put(num, numMap.get(num) + 1);
            } else {
                numMap.put(num, 1);
            }
        }
        int max = 0;
        for (Map.Entry<Integer, Integer> entry : numMap.entrySet()) {
            max = Math.max(max, entry.getValue());
        }

        List<Integer>[] lists = new List[max + 1];
        for (Map.Entry<Integer, Integer> entry : numMap.entrySet()) {
            int count = entry.getValue();
            if (lists[count] == null) {
                lists[count] = new ArrayList<>();
            }
            lists[count].add(entry.getKey());
        }

        List<Integer> topKFrequent = new ArrayList<>();
        for (int i = max; i > 0; --i) {
            if (lists[i] != null) {
                topKFrequent.addAll(lists[i]);
            }
            if (topKFrequent.size() == k) break;
        }
        return topKFrequent;
    }


    public int rotatedDigits(int N) {
        Set<Integer> rotable = new HashSet<>(Arrays.asList(0,1,2,5,6,8,9));
        int count = 0;
        for (int i=1; i<=N; i++) {
            int status = 1;
            for (int j=i; j>0; j/=10) {
                int digit = j%10;
                if (!rotable.contains(digit)) {
                    status = 0;
                    break;
                } else if (status != 2 && (digit == 2 || digit == 5 || digit == 6 || digit == 9)) {
                    status = 2;
                }
            }
            if (status == 2) {
                count++;
            }
        }
        return count;
    }


    public int findLUSlength(int a, String b) {
        return 0;
    }


    public int findKthLargest(int[] nums, int k) {

        int l = 0;
        int r = nums.length-1;

        while (r-l+1 >= k) {
            int pivot = l-1;
            for (int i=l; i<r; i++) {
                if (nums[i] <= nums[r]) {
                    pivot++;
                    int temp = nums[pivot];
                    nums[pivot] = nums[i];
                    nums[i] = temp;
                }
            }
            pivot++;
            int temp = nums[pivot];
            nums[pivot] = nums[r];
            nums[r] = temp;
            int pos = r - pivot + 1;
            if (pos == k) {
                return nums[pivot];
            } else if (pos > k) {
                l = pivot+1;
            } else {
                k -= pos;
                r = pivot-1;

            }

        }
        return -1;
    }


    //search in a rotated sorted array
    public int search2(int[] nums, int target) {
        int l = 0;
        int r = nums.length -1;
        while(l <= r) {
            int mid = l + (r-l)/2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[l] < nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid-1;
                } else {
                    l = mid+1;
                }
            } else if (nums[l] > nums[mid]) {
                if (nums[mid] < target && target <= nums[r]) {
                    l = mid+1;
                } else {
                    r = mid-1;
                }
            } else {
                l++;
            }
        }
        return -1;
    }


    public boolean search1(int[] nums, int target) {
        int l = 0;
        int r = nums.length -1;
        while(l <= r) {
            int mid = l + (r-l)/2;
            if (nums[mid] == target) {
                return true;
            } else if (nums[l] == nums[mid]) {
                l++;
            } else if (nums[l] < nums[mid]) {
                if (nums[l] <= target && target < nums[mid]) {
                    r = mid-1;
                } else {
                    l = mid+1;
                }
            } else {
                if (nums[mid] < target && target <= nums[r]) {
                    l = mid+1;
                } else {
                    r = mid-1;
                }
            }
        }
        return false;
    }


    public boolean judgeCircle(String moves) {
        return true;
    }


    public boolean rotateString(String A, String B) {
        return (A.length() == B.length() && (A+A).contains(B));
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


    public int findMin(int[] nums) {
        int l = 0;
        int r = nums.length - 1;
        while (l <= r) {
            while (nums[l] == nums[r] && l != r) {
                l++;
            }
            if (nums[l] <= nums[r]) {
                return nums[l];
            }
            int mid = l + (r-l)/2;
            if (nums[l] <= nums[mid]) {
                l = mid+1;
            } else {
                r = mid;
            }
        }
        return -1;
    }


    public int kthSmallest(int[][] matrix, int k) {
        int len = matrix.length;
        int low = matrix[0][0];
        int up = matrix[len-1][len-1];

        while(low < up) {
            int mid = low + (up - low)/2;
            if (count(matrix, mid) < k) {
                low = mid+1;
            } else {
                up = mid;
            }
        }
        return up;
    }


    public int count(int[][] matrix, int target) {
        int i = matrix.length -1;
        int j = 0;
        int len = matrix[0].length;
        int count = 0;

        while (i >= 0 && j < len) {
            if (target >= matrix[i][j]) {
                count += i+1;
                j++;
            } else {
                i--;
            }
        }
        return count;
    }


    public int numSquares(int n) {
        if (is_square(n)) {
            return 1;
        }

        int num = n;
        while ((num & 3) ==0) {
            num >>= 2;
        }
        if ((num & 7) == 7) {
            return 4;
        }

        int sqrt = (int)Math.sqrt(n);
        for (int i=1; i<sqrt; i++) {
            if (is_square(n - i*i)) {
                return 2;
            }
        }

        return 3;
    }


    public boolean is_square(int n) {
        int sqrt = (int)Math.sqrt(n);
        return sqrt*sqrt == n;
    }


    public int[] intersect1(int[] nums1, int[] nums2) {
        return nums1;
    }


    public int findDuplicate(int[] nums) {
        return 0;
    }


    public String optimalDivision(int[] nums) {
        return "gghg";
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
        String[] numStrArray = new String[nums.length];
        for (int i=0; i<nums.length; i++) {
            numStrArray[i] = Integer.toString(nums[i]);
        }

        Arrays.sort(numStrArray, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (o2+o1).compareTo(o1+o2);
            }
        });

        StringBuilder sb = new StringBuilder();
        for (String s : numStrArray) {
            sb.append(s);
        }

        while (sb.charAt(0) == '0' && sb.length() > 1) {
            sb.deleteCharAt(0);
        }

        return sb.toString();
    }


    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return -1;
        }
        int max = nums[0];
        int maxRecent = nums[0];
        for (int i=1; i<nums.length; i++) {
            maxRecent = Math.max(maxRecent+nums[i], nums[i]);
            max = Math.max(max, maxRecent);
        }

        return max;
    }


    public void sortColors(int[] nums) {

    }


    public boolean judgeSquareSum(int c) {
        int up = (int)Math.sqrt(c);
        int low = 0;
        while (low <= up) {
            int sum = low*low + up*up;
            if (sum == c) {
                return true;
            } else if (sum < c) {
                low++;
            } else {
                up--;
            }
        }
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


    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        return nums1;
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
        int commonBits;
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
        int diff;
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
        Set<Integer> numbers = new HashSet<>(Arrays.asList(0, 2, 4, 16, 20, 37, 42, 58, 89, 145));
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


    public List<Integer> majorityElementII(int[] nums) {
        return new ArrayList<>();
    }


    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> indexStack = new Stack<>();
        int[] result = new int[temperatures.length];

        for (int i=temperatures.length-1; i>=0; --i) {
            int val = 0;
            while (!indexStack.isEmpty()) {
                if (temperatures[i] < temperatures[indexStack.peek()]) {
                    val = indexStack.peek() - i;
                    break;
                }
                indexStack.pop();
            }
            result[i] = val;
            indexStack.push(i);
        }
        return result;
    }


    public int numSubarrayProductLessThanK(int[] nums, int k) {
        return k;
    }


    public int monotoneIncreasingDigits(int N) {
        return N;
    }


    public boolean pyramidTransition(String bottom, List<String> allowed) {
        return false;
    }


    static class MyQueue {

        /** Initialize your data structure here. */
        public MyQueue() {

        }

        /** Push element x to the back of queue. */
        public void push(int x) {

        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            return 0;
        }

        /** Get the front element. */
        public int peek() {
            return 0;
        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return true;
        }
    }


    static class MyStack {

        /** Initialize your data structure here. */
        public MyStack() {

        }

        /** Push element x onto stack. */
        public void push(int x) {

        }

        /** Removes the element on top of the stack and returns that element. */
        public int pop() {
            return 0;
        }

        /** Get the top element. */
        public int top() {
            return 1;
        }

        /** Returns whether the stack is empty. */
        public boolean empty() {
            return false;
        }
    }


    static class MinStack {

        /** initialize your data structure here. */
        public MinStack() {

        }

        public void push(int x) {

        }

        public void pop() {

        }

        public int top() {
            return 0;
        }

        public int getMin() {
            return 1;
        }
    }
}
