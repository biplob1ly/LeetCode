package data_structure;

/**
 * Created by Biplob on 11/1/2017.
 */
public class SimpleStack {
    private Node top;

    public boolean isEmpty() {
        return top == null;
    }

    public void push(int data) {
        Node node = new Node(data);
        node.next = top;
        top = node;
    }

    public int peek() {
        if (!isEmpty()) {
            return top.data;
        }

        return Integer.MIN_VALUE;
    }

    public int pop() {
        if (!isEmpty()) {
            Node node = top;
            top = top.next;
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
