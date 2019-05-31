/*
 * contlayer.h
 *
 * Container layer contains over layers
 */

#ifndef CONTLAYER_H_
#define CONTLAYER_H_

#include <memory>
#include <vector>

#include "nnlayer.h"

namespace yann {

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::ContainerLayer: consists of N frames (layers)
//
class ContainerLayer : public Layer
{
  typedef Layer Base;

public:
  typedef std::vector<std::unique_ptr<Layer>> Layers;

public:
  ContainerLayer();
  virtual ~ContainerLayer();

  size_t get_layers_num() const     { return _layers.size(); }

public:
  const Layer * get_layer(const size_t & pos) const;
  Layer * get_layer(const size_t & pos);
  virtual void append_layer(std::unique_ptr<Layer> layer);

  // Layer overwrites
  virtual bool is_valid() const;
  virtual bool is_equal(const Layer& other, double tolerance) const;

  virtual void init(enum InitMode mode);

  virtual void read(std::istream & is);
  virtual void write(std::ostream & os) const;

protected:
  const Layers & layers() const { return _layers; }
  Layers & layers()             { return _layers; }

private:
  Layers _layers;
}; // class ContainerLayer

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::SequentialLayer: 1 input gotes through N layers to 1 output
//
class SequentialLayer : public ContainerLayer
{
  typedef ContainerLayer Base;

public:
  SequentialLayer();
  virtual ~SequentialLayer();

public:
  // ContainerLayer overwrites
  virtual void append_layer(std::unique_ptr<Layer> layer);

  virtual void print_info(std::ostream & os) const;
  virtual std::string get_name() const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(
      const MatrixSize & batch_size,
      const std::unique_ptr<Updater> & updater) const;
  virtual std::unique_ptr<Context> create_training_context(
      const RefVectorBatch & output,
      const std::unique_ptr<Updater> & updater) const;

  virtual void feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode = Operation_Assign) const;
  virtual void backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input, Context * context) const;

  virtual void update(Context * context, const size_t & batch_size);
}; // class SequentialLayer

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::MappingLayer: maps N input frames to M layers / outputs.
//
class MappingLayer : public ContainerLayer
{
  typedef ContainerLayer Base;

public:
  typedef std::vector<size_t> InputsMapping; // which inputs map to the given output

public:
  MappingLayer(const size_t & input_frames);
  virtual ~MappingLayer();

  inline size_t get_input_frames_num() const  { return _input_frames; }

public:
  // ContainerLayer overwrites
  virtual void append_layer(std::unique_ptr<Layer> layer, const InputsMapping & mapping);
  virtual void append_layer(std::unique_ptr<Layer> layer);

  virtual bool is_equal(const Layer& other, double tolerance) const;
  virtual void print_info(std::ostream & os) const;
  virtual std::string get_name() const;
  virtual bool is_valid() const;
  virtual MatrixSize get_input_size() const;
  virtual MatrixSize get_output_size() const;

  virtual std::unique_ptr<Context> create_context(const MatrixSize & batch_size) const;
  virtual std::unique_ptr<Context> create_context(const RefVectorBatch & output) const;
  virtual std::unique_ptr<Context> create_training_context(
      const MatrixSize & batch_size,
      const std::unique_ptr<Updater> & updater) const;
  virtual std::unique_ptr<Context> create_training_context(
      const RefVectorBatch & output,
      const std::unique_ptr<Updater> & updater) const;

  virtual void feedforward(const RefConstVectorBatch & input, Context * context, enum OperationMode mode = Operation_Assign) const;
  virtual void backprop(const RefConstVectorBatch & gradient_output, const RefConstVectorBatch & input,
                        boost::optional<RefVectorBatch> gradient_input, Context * context) const;
  virtual void update(Context * context, const size_t & batch_size);

private:
  template<typename InputType>
  InputType get_input_block(InputType input, const size_t & input_frame, const MatrixSize & input_size) const;
  template<typename OutputType>
  OutputType get_output_block(OutputType output, MatrixSize & output_pos, const MatrixSize & output_size) const;

private:
  size_t _input_frames;
  std::vector<InputsMapping> _mappings; // for each output, list the inputs mappings
}; // class MappingLayer

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::BroadcastLayer: send the 1 input to N outputs:
//
class BroadcastLayer : public MappingLayer
{
  typedef MappingLayer Base;

public:
  BroadcastLayer();
  virtual ~BroadcastLayer();

public:
  // MappingLayer overwrites
  virtual std::string get_name() const;
  virtual void append_layer(std::unique_ptr<Layer> layer);
}; // class BroadcastLayer


////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::ParallelLayer: N inputs to N outputs
//
class ParallelLayer : public MappingLayer
{
  typedef MappingLayer Base;

public:
  ParallelLayer(const size_t & input_frames);
  virtual ~ParallelLayer();

public:
  // Overwrites
  virtual std::string get_name() const;
  virtual void append_layer(std::unique_ptr<Layer> layer);
}; // class ParallelLayer

////////////////////////////////////////////////////////////////////////////////////////////////
//
// yann::MergeLayer: N inputs to 1 output
//
class MergeLayer : public MappingLayer
{
  typedef MappingLayer Base;

public:
  MergeLayer(const size_t & input_frames);
  virtual ~MergeLayer();

public:
  // Overwrites
  virtual std::string get_name() const;
  virtual void append_layer(std::unique_ptr<Layer> layer);
}; // class MergeLayer


}; // namespace yann

#endif /* CONTLAYER_H_ */
