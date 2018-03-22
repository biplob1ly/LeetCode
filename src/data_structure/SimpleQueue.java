package data_structure;

/**
 * Created by Biplob on 11/3/2017.
 */
public class SimpleQueue {
    private Node head;
    private Node tail;

    public boolean isEmpty() {
        return head == null;
    }

    public void push(int data) {
        Node node = new Node(data);
        if (isEmpty()) {
            head = tail = node;
            return;
        } else {
            tail.next = node;
            tail = tail.next;
            return;
        }
    }

    public int peek() {
        if (!isEmpty()) {
            return head.data;
        }

        return Integer.MIN_VALUE;
    }

    public int pop() {
        if (!isEmpty()) {
            Node node = head;
            head = head.next;
            if (head == null) {
                tail = null;
            }
            return node.data;
        }

        return Integer.MIN_VALUE;
    }

    class Node {
        int data;
        Node next;

        public Node(int data) {
            this.data = data;
        }
    }
}
