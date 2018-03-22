package algorithms;

/**
 * Created by Biplob on 10/18/2017.
 */
public class Searching {

    public static int binarySearchR(int[] arr, int x) {
        return binarySearchRecursive(arr, x, 0, arr.length-1);
    }

    public static int binarySearchRecursive(int[] arr, int x, int start, int end) {
        if (start > end) {
            return -1;
        }

        int mid = start + (end - start)/2;

        if (x == arr[mid]) {
            return mid;
        } else if (x < arr[mid]) {
            return binarySearchRecursive(arr, x, start, mid-1);
        } else {
            return binarySearchRecursive(arr, x, mid+1, end);
        }
    }

    public static int binarySearchI(int[] arr, int x) {
        return binarySearchIterative(arr, x, 0, arr.length-1);
    }

    public static int binarySearchIterative(int[] arr, int x, int start, int end) {
        while (start <= end) {
            int mid = start + (end - start)/2;
            if (x == arr[mid]) {
                return mid;
            } else if (x < arr[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return -1;
    }
}
