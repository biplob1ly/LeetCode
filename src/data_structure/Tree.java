package data_structure;

/**
 * Created by Biplob on 11/8/2017.
 */
public class Tree {
    Node root;

    public Node search(int data) {
        Node node = root;
        while(node != null && node.data != data) {
            if (data < node.data) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        return node;
    }

    public Node insert(int data) {
        Node node = new Node(data);
        if (root == null) {
            return root = node;
        }
        Node parent = root;
        while (parent != null) {
            if (data < parent.data) {
                if (parent.left == null) {
                    parent.left = node;
                    break;
                }
                parent = parent.left;
            } else {
                if (parent.right == null) {
                    parent.right = node;
                    break;
                }
                parent = parent.right;
            }
        }
        return root;
    }

    public Node getInorderSuccessor(Node node) {
        Node parent = node.right;
        while (parent.left != null) {
            parent = parent.left;
        }
        return parent;
    }

    public Node delete(int data) {
        Node node = search(data);
        if (node.left != null && node.right != null) {
            Node successor = getInorderSuccessor(node);
            node.data = successor.data;
            successor = successor.right;
        } else if (node.left != null) {

        } else {

        }
        return root;
    }

    class Node {
        Node parent;
        Node left,right;
        int data;

        public Node(int data) {
            this.data = data;
        }
    }
}
