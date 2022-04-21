# How to determine what is captured in PCA of an MD trajectory

Contains a file which allows users to take a molecular dynamics (MD) trajectory of e.g. a protein and perform principal component analysis (PCA) on the trajectory to determine which motions are the most 'important' (i.e., which parts of the protein move the most).

For more informatio about PCA, eigenvalues, etc., see [this link.](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)

Once PCA has been performed, users can see what motions are being captured along each PC by keeping all other eigenvalues the same, and just altering the eigenvalue of a specific prinicipal coordinate. 

![pca_graph](https://user-images.githubusercontent.com/42864940/164477110-8ede716b-0f23-4ab5-8764-86c6c8930207.png)

For example, the figure above shows a trajectory that captures drug unbinding from a protein, projected in PC space. In order to see what motions are being captured along PC 1, we keep the eigenvalues for each structure along PC 2 the same and increase the value along PC 1. 

From this new transformed data, we can backtransform the data and get out the distances and corresponding cooefficients for each new altered eigenvalue. In simpler terms, for each new structure along PC 1 (where we have changed the eigenvalue), we can calculate every distance (and the coefficient, which is just a measure of how much that distance contributes to the PC) that would make this new structure. The result is that we can create a structure (in terms of the atomic distances) for what is being captured as we move along each PC. See the below figure, where we alter PC 1 (and keep other PC values the same), and we have a new structure that corresponds to the change along PC1. 

![alter_eigenvalues](https://user-images.githubusercontent.com/42864940/164478315-744ccc76-7f50-4747-9ceb-0ec8a3a721b2.png)
