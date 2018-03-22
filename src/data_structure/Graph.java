package data_structure;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by Biplob on 10/23/2017.
 */
public class Graph {

    private HashMap<Integer, Node> nodeMap;
    
    public Graph(String inputFileName) {
        nodeMap = new HashMap<>();
        readFromFile(inputFileName);
        for (Map.Entry<Integer, Node> pair : nodeMap.entrySet()) {
            System.out.println(pair.getValue().toString());
        }
    }

    private void readFromFile(String fileName) {
        BufferedReader br = null;
        FileReader fr = null;

        try {
            br = new BufferedReader(new FileReader(fileName));
            String line;
            boolean isFirstLine= true;
            while((line = br.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(line, " ");
                if (isFirstLine) {
                    while (st.hasMoreTokens()) {
                        int id = Integer.parseInt(st.nextToken());
                        nodeMap.put(id, new Node(id));
                    }
                    isFirstLine= false;
                }else {
                    Node parentNode = nodeMap.get(Integer.parseInt(st.nextToken()));
                    while (st.hasMoreTokens()) {
                        int id = Integer.parseInt(st.nextToken());
                        parentNode.addEdge(id);
                    }
                }

            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        } catch (IOException e) {
            System.out.println("IO exception");
        } finally {
            try {
                if (fr != null) {
                    fr.close();
                }

                if (br != null) {
                    br.close();
                }
            }
            catch (IOException e) {
                System.out.println("IO exception");
            }
        }
    }

    public void traverseDFS() {
        HashSet<Integer> visitedNodeId = new HashSet<>();

        for (int nodeId : nodeMap.keySet()) {
            if (!visitedNodeId.contains(nodeId)) {
                DFSUtil(nodeId, visitedNodeId);
                System.out.println();
            }
        }
    }

    private void DFSUtil(int nodeId, HashSet<Integer> visitedNodeId) {
        if (!visitedNodeId.contains(nodeId)) {
            visitedNodeId.add(nodeId);
            System.out.print(nodeId + " ");
            for (int childId : nodeMap.get(nodeId).getchildIdList()) {
                DFSUtil(childId, visitedNodeId);
            }
        }
    }

    public void traverseBFS() {
        LinkedList<Integer> queue = new LinkedList<>();
        HashSet<Integer> visitedNodeId = new HashSet<>();
        //add source
        visitedNodeId.add(1);
        queue.add(1);

        while (!queue.isEmpty()) {
            Node current = nodeMap.get(queue.remove());
            System.out.print(current.getId() + " ");
            for (int childId : current.getchildIdList()) {
                if (!visitedNodeId.contains(childId)) {
                    visitedNodeId.add(childId);
                    queue.add(childId);
                }
            }
        }
    }

    class Node {
        private int id;
        private LinkedList<Integer> childIdList;
        public Node(int id) {
            this.id = id;
            childIdList = new LinkedList<>();
        }

        public void addEdge(int nodeId) {
            childIdList.add(nodeId);
        }

        public int getId() {
            return id;
        }

        public LinkedList<Integer> getchildIdList() {
            return childIdList;
        }

        @Override
        public String toString() {
            String outStr = id + "";
            for (int childId : childIdList) {
                outStr += " " + childId;
            }
            return outStr;
        }
    }
}

