# Genetic Algorithm for Maximizing a Function

import random

DEFAULT_GENE_LENGTH = 5
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GENERATION_COUNT = 50

MUTATION_RATE = 0.1

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
        self.genes[index] = random.randint(0, 1)
        self.fitness = self.calculate_fitness()

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}, Integer: {self.to_integer(self.genes)}"
    
class Population:
    def __init__(self, size=DEFAULT_POPULATION_SIZE, gene_length=DEFAULT_GENE_LENGTH, fitness_function=sum):
        self.chromosomes = [Chromosome([random.randint(0, 1) for _ in range(gene_length)], fitness_function) for _ in range(size)]
        self.best_chromosome = self.get_best_chromosome()

    def evolve(self):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        parents = self.chromosomes[:len(self.chromosomes)//2]  # Select the top half

        next_generation = []

        while len(next_generation) < len(self.chromosomes):
            parent1, parent2 = random.choices(parents, k=2)
            
            child1, child2 = self.crossover(parent1, parent2) 

            if random.random() < MUTATION_RATE:
                child1.mutate()

            if random.random() < MUTATION_RATE:
                child2.mutate()

            next_generation.append(child1)
            next_generation.append(child2)

        self.chromosomes = next_generation
        best = self.get_best_chromosome()
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best


    def crossover(self, parent1, parent2, crossover_point=None):
        if crossover_point is None:
            crossover_point = random.randint(1, len(parent1.genes) - 1)

        child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

        return Chromosome(child1_genes, parent1.fitness_function), Chromosome(child2_genes, parent2.fitness_function)

    def get_best_chromosome(self):
        return max(self.chromosomes, key=lambda x: x.fitness)

    def __str__(self):
        return '\n'.join(str(chromosome) for chromosome in self.chromosomes)
    
def example(genes):
    x = int(''.join([str(bit) for bit in genes]), 2)
    return -x**2/10 + 3*x
    
if(__name__ == "__main__"):
    population = Population(fitness_function=example)

    print("Initial Population:")
    print(population)
    print(f"Best Chromosome: {population.best_chromosome}")
    print()

    for generation in range(DEFAULT_GENERATION_COUNT):
        population.evolve()
        print(f"Generation {generation + 1}: Best Chromosome: {population.best_chromosome}")
    
    print(f"\nFinal Best Chromosome: {population.best_chromosome}")
