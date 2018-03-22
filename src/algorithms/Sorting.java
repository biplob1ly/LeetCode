package algorithms;

/**
 * Created by Biplob on 10/18/2017.
 */
public class Sorting {

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    public static void bubbleSort(int[] arr) {
        boolean isSorted = false;
        int sortedUpto = arr.length -1;
        while(!isSorted) {
            isSorted = true;
            for (int i=0; i<sortedUpto; i++) {
                if (arr[i] > arr[i+1]) {
                    swap(arr, i, i+1);
                    isSorted = false;
                }
            }
            sortedUpto--;
        }
    }

    public static void mergeSort(int[] arr) {
        mergeRecursively(arr, 0, arr.length - 1);
    }

    public static void mergeRecursively(int[] arr, int start, int end) {
        if (start < end) {
            int mid = (start + end)/2;
            mergeRecursively(arr, start, mid);
            mergeRecursively(arr, mid+1, end);
            merge(arr, start, mid, end);
        }
    }

    public static void merge(int[] arr, int start, int mid, int end) {
        int n1 = mid - start +1;
        int n2 = end - mid;
        int[] left = new int[n1];
        int[] right = new int[n2];
        System.arraycopy(arr, start, left, 0, n1);
        System.arraycopy(arr, mid+1, right, 0, n2);
        int i=0, j=0, k=start;
        while(i<n1 && j<n2) {
            if (left[i] <= right[j]) {
                arr[k++] = left[i++];
            } else {
                arr[k++] = right[j++];
            }
        }
        //Only one of the following lines will work
        System.arraycopy(left, i, arr, k, n1-i);
        System.arraycopy(right, j, arr, k, n2-j);
    }

    public static void quickSort(int[] arr) {
        quickSortRecursively(arr, 0, arr.length - 1);
    }

    public static void quickSortRecursively(int[] arr, int start, int end) {
        if (start < end) {
            int index = partition(arr, start, end);
            quickSortRecursively(arr, start, index-1);
            quickSortRecursively(arr, index+1, end);
        }
    }

    public static int partition(int[] arr, int start, int end) {
        int pivot = arr[end];
        int i = start-1;
        for (int j=start; j<end; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        swap(arr, i+1, end);
        return i+1;
    }
}
