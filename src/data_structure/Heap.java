package data_structure;

import java.util.Arrays;

/**
 * Created by Biplob on 11/8/2017.
 */
public class Heap {
    private int capacity;
    private int size;
    private int[] items;

    public Heap(int capacity) {
        this.capacity = capacity;
        items = new int[capacity];
    }

    private int getParentIndex(int index) {
        return (index-1)/2;
    }

    private int getLeftChildIndex(int index) {
        return 2*index+1;
    }

    private int getRightChildIndex(int index) {
        return 2*index+2;
    }

    private boolean hasParent(int index) {
        return getParentIndex(index) >= 0;
    }

    private boolean hasLeftChild(int index) {
        return getLeftChildIndex(index) < size;
    }

    private boolean hasRightChild(int index) {
        return getRightChildIndex(index) < size;
    }

    private void swap(int index1, int index2) {
        int temp = items[index1];
        items[index1] = items[index2];
        items[index2] = temp;
    }

    private void ensureCapacity() {
        if (size == capacity) {
            capacity = 2*capacity;
            items = Arrays.copyOf(items, capacity);
        }
    }

    public void add(int val) {
        ensureCapacity();
        items[size++] = val;
        heapifyUp();
    }

    public int peek() {
        if (size == 0) {
            throw new IllegalStateException("Heap is empty");
        }
        return items[0];
    }

    public int poll() {
        if (size == 0) {
            throw new IllegalStateException("Heap is empty");
        }
        int item = items[0];
        items[0] = items[size-1];
        items[size-1] = 0;
        size--;
        heapifyDown();
        return item;
    }

    private void heapifyUp() {
        int index = size - 1;
        while (hasParent(index) && items[getParentIndex(index)] > items[index]) {
            swap(getParentIndex(index), index);
            index = getParentIndex(index);
        }
    }

    private void heapifyDown() {
        int index = 0;
        while (hasLeftChild(index)) {
            int smallerChildIndex = getLeftChildIndex(index);
            if (hasRightChild(index) && items[getRightChildIndex(index)] < items[smallerChildIndex]) {
                smallerChildIndex = getRightChildIndex(index);
            }
            if (items[index] > items[smallerChildIndex]) {
                swap(smallerChildIndex, index);
            } else {
                break;
            }
            index = smallerChildIndex;
        }
    }

    @Override
    public String toString() {
        return Arrays.toString(items);
    }
}
