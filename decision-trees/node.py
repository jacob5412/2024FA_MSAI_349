"""
A node class.
"""


class Node:
    """
    Represents a node in a ID3 decision tree.

    Attributes:
        attribute (str): The attribute on which the data is split.
        attribute_value (str): The value of the attribute for this node.
        class_label (str): The class label if it's leaf node, otherwise
                           the majority class if it's a non-leaf node.
        children (dict of Node): Dictionary of child nodes of the
                                 form {attribute_value:child_node}.
        is_leaf (bool): True if it's a leaf node, False otherwise.
    """

    def __init__(
        self,
        attribute=None,
        attribute_value=None,
        class_label=None,
        is_leaf=False,
    ):
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.class_label = class_label
        self.children = {}
        self.is_leaf = is_leaf

    def has_children(self):
        """
        Returns True if current node has children, False otherwise.
        """
        return bool(self.children)

    def get_children(self):
        """
        Returns children of current node.
        """
        return self.children.items()

    def add_child(self, attribute_value, child_node):
        """
        Add a child node to the current node with a given attribute value.
        """
        self.children[attribute_value] = child_node

    def get_attribute(self):
        """
        Returns attribute of current node.
        """
        return self.attribute

    def update_attribute(self, attribute):
        """
        Updates attribute of current node.
        """
        self.attribute = attribute

    def update_as_leaf(self, is_leaf=True):
        """
        Update current node to a leaf node.
        """
        self.is_leaf = is_leaf

    def update_class_label(self, class_label):
        """
        Update class label of current node.
        """
        self.class_label = class_label

    def display(self, indent=0):
        """
        (Debugging) Displays a node and its children.
        """
        prefix = "  " * indent
        if self.is_leaf:
            print(f"| {prefix}└─Class: {self.class_label}")
        else:
            print(
                f"{'|' if indent != 0 else ''}{prefix}"
                + f"{'└─' if indent != 0 else ''}"
                + f"Attribute: {self.get_attribute()}"
            )
            for value, child_node in self.get_children():
                print(f"|{prefix} └─Value {value}:")
                child_node.display(indent + 1)
