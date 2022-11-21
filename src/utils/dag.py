from typing import Optional
import random


class DAG(object):
    class Node(object):
        def __init__(self, topo_id: int, is_input: bool = False, is_output: bool = False,
                     output_shape: Optional[tuple] = None):
            self.id = topo_id
            self.is_input = is_input
            self.is_output = is_output
            self.output_shape = output_shape
            self.type = None

            self.inbound_nodes = []
            self.outbound_nodes = []

        # Hong: Property decorator here to allow node_object.is_merging
        @property
        def is_merging(self):
            return len(self.inbound_nodes) > 1

        # Hong: Property decorator here to allow node_object.is_connected
        @property
        def is_connected(self):
            return bool(self.inbound_nodes)

        def remove(self):
            # Hong: Why we only consider the first inbound node????
            n1 = self.inbound_nodes[0]
            n1.connect_to(self.outbound_nodes)
            # Hong: Should we remove self from n1 outbound nodes as well?
            # n1.outbound_nodes.remove(self)
            for n2 in self.outbound_nodes:
                # print(f"{self.id} -x-> {n2.id}")
                n2.inbound_nodes.remove(self)

        def connect_to(self, nodes):
            for n2 in nodes:
                # print(f"{self.id} --> {n2.id}")
                n2.inbound_nodes.append(self)
                self.outbound_nodes.append(n2)

    def __init__(self, main_node_num: int, input_shapes: list, output_shapes: list, max_branch_num: int):
        """
        Initialize the DAG
        :param main_node_num: number of nodes
        :param input_shapes: a list of shapes of input nodes
        :param output_shapes: a list of shapes of output nodes
        :param max_branch_num:
        """
        if len(input_shapes) + len(output_shapes) > main_node_num:
            raise ValueError("Dag's node num is not enough.")

        self.__max_branch_num = max_branch_num

        # Select input and output nodes
        # Hong: First node is input and the last node is output
        sampled_id = random.sample(range(1, main_node_num-1), k=len(input_shapes)+len(output_shapes)-2)
        random.shuffle(sampled_id)
        inputs_id = sorted([0] + sampled_id[:len(input_shapes)-1])
        outputs_id = sorted(sampled_id[len(input_shapes)-1:] + [main_node_num-1])
        # print(f"inputs_id: {inputs_id}  outputs_id: {outputs_id}")
        self.nodes = []
        for i, input_shape in zip(inputs_id, input_shapes):
            self.nodes.append(self.Node(i, is_input=True, output_shape=input_shape))
        for i, output_shape in zip(outputs_id, output_shapes):
            self.nodes.append(self.Node(i, is_output=True, output_shape=output_shape))
        for i in range(main_node_num):
            if i not in inputs_id and i not in outputs_id:
                self.nodes.append(self.Node(i))
        self.nodes.sort(key=lambda n: n.id)  # sort by the node id
        self.__generate()
        self.show()

    def __generate(self, adjoin_prob: float = 0.9):
        # For each non-output node, connect to at least one node backward\
        # the target cannot be an input node
        for cur_id, n1 in enumerate(self.nodes):
            if not n1.is_output:
                if not self.nodes[cur_id+1].is_input and random.random() < adjoin_prob:  # With high prob connecting to the next node
                    targets = [self.nodes[cur_id+1]]
                else:
                    targets = [n2 for n2 in self.nodes[cur_id+1:] if not n2.is_input]
                n1.connect_to(random.sample(targets, random.randint(1, min(len(targets), self.__max_branch_num-1))))
        # For each non-input node that have no inbound node, let it be connected to one previous (Make sure all output node are connected)
        for cur_id, n2 in enumerate(self.nodes):
            if not n2.is_input and not n2.is_connected:
                if not self.nodes[cur_id-1].is_output and random.random() < adjoin_prob:  # With high prob connecting to the previous node
                    targets = [self.nodes[cur_id-1]]
                else:
                    targets = [n1 for n1 in self.nodes[:cur_id] if not n1.is_output]
                random.choice(targets).connect_to([n2])

    def show(self):
        for n1 in self.nodes:
            print(f"n1: {n1.id}  n2: {[n2.id for n2 in n1.outbound_nodes]}")


if __name__ == '__main__':
    dag = DAG(100, [(32, 32, 4), (3, 2)], [(3, 5, 6)], 4)
