require 'matrix'

class Var
  def initialize(shape: Array, desc: 'Variable')
    @shape = shape
    @desc = desc
    @value = Matrix.zero(*@shape)
    pp "Initialized #{@desc} with shape #{@shape}"
  end

  def value
    @value
  end

end




def initialize_simplex(_A, b, c)
  return 1, 2, 3, 4, 5 ,6
end

def simplex(_A, b, c)
  _N, _B, _A, b, c, v = initialize_simplex(_A,b,c)
  d = Matrix.build(c.row_count,0) {0}
  while test

  end
end
