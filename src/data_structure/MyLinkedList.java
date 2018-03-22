package data_structure;

/**
 * Created by Biplob on 10/31/2017.
 */
public class MyLinkedList {

    private Node head;

    public MyLinkedList(int[] array) {
        for (int num : array) {
            prepend(num);
        }
    }

    public int search(int data) {
        int index = -1;
        Node current = head;
        while (current != null) {
            index++;
            System.out.print(current.data + " ");
            if (current.data == data) {
                System.out.println();
                return index;
            }
            current = current.next;
        }
        System.out.println();
        return -1;
    }

    public void appendNode(int data) {
        if (head == null) {
            head = new Node(data);
        } else {
            Node current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = new Node(data);
        }
    }

    public void prepend(int data) {
        if (head == null) {
            head = new Node(data);
        } else {
            Node newNode = new Node(data);
            newNode.next = head;
            head = newNode;
        }
    }

    public void deleteNode(int data) {
        if (head != null) {
            if (head.data == data) {
                head = head.next;
            }
            Node current = head;
            while (current.next != null) {
                if (current.next.data == data) {
                    current.next = current.next.next;
                    return;
                }
                current = current.next;
            }
        }
    }

    class Node {
        int data;
        Node next;

        public Node(int data) {
            this.data = data;
        }
    }
}
