import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import simple_norm

def log_figure(img,fig_name,experiment,cmap="plasma"):
	f, ax = plt.subplots()


	im = ax.imshow(
	    img, 
	    cmap="plasma", # or whatever colormap you want
	    origin="lower",
	        #stretch can be also be {‘linear’, ‘sqrt’, ‘power’, log’, ‘asinh’}
	    norm=simple_norm(img, stretch="linear")
	)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	f.colorbar(im, cax=cax, orientation='vertical')

	ax.set_xticks([])
	ax.set_yticks([])
	plt.tight_layout()

	experiment.log_figure(
	    figure_name=fig_name,
	    figure=f,
	)

	# do not forget to close the figure
	plt.close()