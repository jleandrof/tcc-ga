# GAADT implementation for TSP

import random
from sys import maxsize as MAX_INT
from collections import deque

DEFAULT_GENE_LENGTH = 10
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GENERATION_COUNT = 20


MUTATION_RATE = 0.1

CITIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DISTANCE_MATRIX = {
    (1, 2): 10, (1, 3): 15, (1, 4): 20, (1, 5): 25, (1, 6): 30, (1, 7): 35, (1, 8): 40, (1, 9): 45, (1, 10): 50,
    (2, 3): 35, (2, 4): 25, (2, 5): 30, (2, 6): 20, (2, 7): 15, (2, 8): 40, (2, 9): 45, (2, 10): 50, (2, 1): 100,
    (3, 4): 30, (3, 5): 20, (3, 6): 25, (3, 7): 35, (3, 8): 40, (3, 9): 45, (3, 10): 50, (3, 1): 100, (3, 2): 95,
    (4, 5): 15, (4, 6): 30, (4, 7): 25, (4, 8): 35, (4, 9): 40, (4, 10): 45, (4, 1): 100, (4, 2): 95, (4, 3): 90,
    (5, 6): 20, (5, 7): 30, (5, 8): 25, (5, 9): 35, (5, 10): 40, (5, 1): 100, (5, 2): 95, (5, 3): 90, (5, 4): 85,
    (6, 7): 15, (6, 8): 20, (6, 9): 30, (6, 10): 35, (6, 1): 100, (6, 2): 95, (6, 3): 90, (6, 4): 85, (6, 5): 80,
    (7, 8): 10, (7, 9): 25, (7, 10): 30, (7, 1): 100, (7, 2): 95, (7, 3): 90, (7, 4): 85, (7, 5): 80, (7, 6): 75,
    (8, 9): 15, (8, 10): 20, (8, 7): 100,
    (9, 10): 10, (9, 1): 55, (9, 2): 50, (9, 3): 45, (9, 4): 40, (9, 5): 35, (9, 6): 30, (9, 7): 25, (9, 8): 20,
    (10, 1): 55, (10, 2): 50, (10, 3): 45, (10, 4): 40, (10, 5): 35, (10, 6): 30, (10, 7): 25, (10, 8): 20, (10, 9): 15
}

children = []
# DISTANCE_MATRIX = {
#     (1, 2): 10, (1, 3): 15, (1, 4): 20, (1, 5): 25,
#     (2, 3): 35, (2, 4): 25, (2, 5): 30,
#     (3, 4): 30, (3, 5): 20,
#     (4, 5): 15
# }
# DISTANCE_MATRIX = {
#     (1, 2): 10, (1, 3): 15, (1, 4): 20, (1, 5): 25,
#     (2, 1): 100, (2, 3): 35, (2, 4): 25, (2, 5): 30,
#     (3, 4): 30, (3, 5): 20,
#     (3, 4): 200, (4, 5): 15
# }

class Chromosome:
    def __init__(self, genes, fitness_function=sum):
        self.genes = genes
        self.fitness_function = fitness_function
        self.fitness = self.calculate_fitness()

    def to_integer(self, genes):
        return int(''.join([str(bit) for bit in genes]), 2)

    def calculate_fitness(self):
        return self.fitness_function(self.genes)

    def mutate(self):
        index = random.randint(0, len(self.genes) - 1)
        city, alternate = random.sample(self.genes, 2)
        self.genes[index] = city if self.genes[index] != city else alternate
        self.fitness = self.calculate_fitness()

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"
    
    def __eq__(self, other):
        return self.genes == other.genes
    
    def __hash__(self):
        return hash(tuple(self.genes))
    
class Population:
    def __init__(self, size=DEFAULT_POPULATION_SIZE, gene_length=DEFAULT_GENE_LENGTH, fitness_function=sum):
        self.chromosomes = [Chromosome(random.sample(CITIES, gene_length), fitness_function) for _ in range(size)]
        # self.chromosomes.append(Chromosome([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fitness_function))
        # self.chromosomes.append(Chromosome([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], fitness_function))
        self.best_chromosome = self.get_best_chromosome()

    def evolve(self):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        parents = self.chromosomes[:len(self.chromosomes)//2]  # Select the top half

        next_generation = []

        for _ in range(len(self.chromosomes)):
            parent1, parent2 = random.choices(parents, k=2)
            
            # child1, child2 = self.crossover(parent1, parent2) 
            child1 = Chromosome(self.crossover_gaadt(parent1, parent2), tsm_fitness)
            child2 = Chromosome(self.crossover_gaadt(parent2, parent1), tsm_fitness)

            if random.random() < MUTATION_RATE:
                child1.mutate()

            if random.random() < MUTATION_RATE:
                child2.mutate()

            children.append((child1, child1.fitness))
            children.append((child2, child2.fitness))

            if self.are_genes_valid(child1.genes):
                next_generation.append(child1)

            if self.are_genes_valid(child2.genes):
                next_generation.append(child2)

            # print(next_generation)

        # next_generation.append(Chromosome([1, 3, 4, 2, 10, 7, 8, 9, 5, 6], tsl_fitness))
        self.chromosomes.extend(next_generation)
        self.chromosomes = list(set(self.chromosomes)) # Remove duplicates
        # print([x.genes for x in self.chromosomes])
        best = self.get_best_chromosome()
        if self.best_chromosome is None or best.fitness < self.best_chromosome.fitness:
            self.best_chromosome = best

        # Population trimming based on average fitness
        # if(len(self.chromosomes) > DEFAULT_POPULATION_SIZE):
        #     average_fitness = sum([chromosome.fitness for chromosome in self.chromosomes]) / len(self.chromosomes)
        #     self.chromosomes = [chromosome for chromosome in self.chromosomes if chromosome.fitness <= average_fitness]


    def crossover(self, parent1, parent2, crossover_point=None):
        # print("p1", parent1.genes, "p2", parent2.genes)
        if crossover_point is None:
            crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        return Chromosome(child1_genes, parent1.fitness_function), Chromosome(child2_genes, parent2.fitness_function)
    
    def crossover_gaadt(self, parent1, parent2):
        genes_1 = parent1.genes

        child = []

        for gene_1 in genes_1:
            try:
                gene_1_next = parent1.genes[parent1.genes.index(gene_1)+1]
                dominant = gene_1_next
                if(gene_1 != parent2.genes[-1]):
                    gene_2, gene_2_next = parent2.genes[parent2.genes.index(gene_1)], parent2.genes[parent2.genes.index(gene_1)+1]
                    if(gene_2_next not in child):
                        dominant = gene_1_next if DISTANCE_MATRIX.get((gene_1, gene_1_next), MAX_INT) < DISTANCE_MATRIX.get((gene_2, gene_2_next), MAX_INT) else gene_2_next

                child.append(dominant)
            except (ValueError, IndexError):
                child.append(random.choice([city for city in CITIES if city not in child]))

        return child
            
    
    def are_genes_valid(self, genes):
        return sorted(genes) == sorted(CITIES[:len(genes)])

    def get_best_chromosome(self):
        return min(self.chromosomes, key=lambda x: x.fitness)

    def __str__(self):
        return '\n'.join(str(chromosome) for chromosome in self.chromosomes)
    
def example(genes):
    x = int(''.join([str(bit) for bit in genes]), 2)
    return -x**2/10 + 3*x

def tsm_fitness(genes):
    total_distance = 0
    for i in range(len(genes) - 1):
        city1, city2 = genes[i], genes[i + 1]
        total_distance += DISTANCE_MATRIX.get((city1, city2), MAX_INT)
    # total_distance += DISTANCE_MATRIX.get((genes[-1], genes[0]), MAX_INT)  # Return to starting city
    return total_distance
    
if(__name__ == "__main__"):
    population = Population(fitness_function=tsm_fitness)

    print("Initial Population:")
    print(population)
    print(f"Best Chromosome: {population.best_chromosome}")
    print()

    initial_best = population.best_chromosome
    for generation in range(DEFAULT_GENERATION_COUNT):
        population.evolve()
        print(f"Generation {generation + 1}: Best Chromosome: {population.best_chromosome}, Population Size: {len(population.chromosomes)}")
    
    print(f"\nInitial Best Chromosome: {initial_best}")
    print(f"Final Best Chromosome: {population.best_chromosome}")
    child_sample = [(child[0].genes, child[1]) for child in children[:10]]
